# Stochastic Optimal Control Fine-Tuning of Stable Diffusion

## Overview
This repository provides an implementation of reward fine-tuning methods for Stable Diffusion 1.5 based on stochastic optimal control (SOC), focusing on [Adjoint Matching](https://openreview.net/forum?id=xQBRrtQM8u). It adds specialized trainers, custom schedulers, and prompt-based dataloaders.

## Features
- Stochastic Optimal Control trainer (SOCTrainer): general purpose class that can be used to define child classes for specific fine-tuning methods 
- Buffer Stochastic Optimal Control trainer (BufferSOCTrainer), which inherits from SOCTrainer and is a general purpose class for training with buffers 
- Adjoint Matching and Buffer Adjoint Matching trainer classes (AMTrainer, BufferAMTRainer), which inherit from SOCTrainer and BufferAMTRainer and implement the adjoint matching algorithm
- SOC DDIM scheduler (SOCDDIMScheduler): augmented version of the Hugging Face Diffusers DDIMScheduler to deal with SOC
- PromptDataModule and BufferPromptDataModule for training and evaluation prompts
- Evaluation scripts for checkpoint metrics

## Quick Start

### Installation
```
git clone https://github.com/microsoft/soc-fine-tuning-sd.git
cd soc-fine-tuning-sd
conda create -n soc-env python=3.10.16
conda activate soc-env
pip install -r requirements.txt
```

### Training
On-policy training:
```
python src/train.py --config configs/multi_prompt.yaml
```
Training with buffer (multiple times faster):
```
python src/train.py --config configs/multi_prompt_buffer.yaml
```

### Evaluation
To evaluate the base model:
```
python src/evaluate_checkpoint.py --num_samples_per_prompt 10 --ckpt base_model --config configs/multi_prompt.yaml
```
To evaluate a checkpoint:
```
python src/evaluate_checkpoint.py --num_samples_per_prompt 10 --ckpt path/to/model.ckpt
```

### Checkpoints and Logs
Saved to the directory set by save_dir in your config.

## Evaluation Metrics for selected checkpoints

The following table summarizes the evaluation metrics for various fine-tuned models using reward multipliers $\lambda \in \{1, 10, 10^{3/2}, 100\}$. The base model refers to the runwayml/stable-diffusion-v1-5 on the Huggingface Model Hub. Each model was evaluated with both $\eta=0.0$ (deterministic sampling) and $\eta=1.0$ (standard noise level, DDPM) at `num_inference_timesteps=50`, and with the fine-tuning scheduler (`beta_start=0.002`, `beta_end=0.009`, `timestep_spacing = 'trailing'`) as well as with the default Huggingface DDIMScheduler settings (`beta_start=0.00085`, `beta_end=0.012`, `timestep_spacing = 'leading'`). The models were fine-tuned using `multi_prompt.yaml`, changing the default `prompt_dropout=0.2` to `prompt_dropout=0.0`. Using `multi_prompt_buffer.yaml` produces similar results. All metrics are reported with standard errors.

### Results under the fine-tuning scheduler
| Model | $\eta$ | ImageReward ↑ | CLIP-Score ↑ | HPS ↑ | DreamSim Var ↑ |
|-------|---------|---------------|-------------|-------|---------------|
| Base model | 0.0 | 0.1873 ± 0.0762 | 0.2746 ± 0.0032 | 0.2566 ± 0.0030 | 0.3849 ± 0.0105 |
| Base model | 1.0 | 0.2801 ± 0.0735 | 0.2766 ± 0.0032 | 0.2560 ± 0.0026 | 0.3471 ± 0.0095 |
| $\lambda = 10^0$ | 0.0 | 0.2170 ± 0.0755 | 0.2754 ± 0.0032 | 0.2576 ± 0.0030 | 0.3826 ± 0.0104 |
| $\lambda = 10^0$ | 1.0 | 0.3057 ± 0.0729 | 0.2767 ± 0.0032 | 0.2565 ± 0.0026 | 0.3446 ± 0.0096 |
| $\lambda = 10^1$ | 0.0 | 0.3419 ± 0.0738 | 0.2764 ± 0.0033 | 0.2623 ± 0.0029 | 0.3774 ± 0.0106 |
| $\lambda = 10^1$ | 1.0 | 0.4134 ± 0.0726 | 0.2781 ± 0.0032 | 0.2608 ± 0.0026 | 0.3408 ± 0.0097 |
| $\lambda = 10^{3/2}$ | 0.0 | 0.5554 ± 0.0730 | 0.2780 ± 0.0033 | 0.2700 ± 0.0028 | 0.3632 ± 0.0108 |
| $\lambda = 10^{3/2}$ | 1.0 | 0.5952 ± 0.0705 | 0.2791 ± 0.0032 | 0.2685 ± 0.0027 | 0.3277 ± 0.0094 |
| $\lambda = 10^2$ | 0.0 | 0.7873 ± 0.0689 | 0.2792 ± 0.0033 | 0.2791 ± 0.0028 | 0.3363 ± 0.0101 |
| $\lambda = 10^2$ | 1.0 | 0.8197 ± 0.0684 | 0.2800 ± 0.0032 | 0.2782 ± 0.0027 | 0.3049 ± 0.0093 |

### Results under the default scheduler
| Model | $\eta$ | ImageReward ↑ | CLIP-Score ↑ | HPS ↑ | DreamSim Var ↑ |
|-------|---------|---------------|-------------|-------|---------------|
| Base model | 0.0 | 0.1262 ± 0.0742 | 0.2771 ± 0.0034 | 0.2417 ± 0.0032 | 0.4025 ± 0.0115 |
| Base model | 1.0 | 0.1938 ± 0.0768 | 0.2794 ± 0.0034 | 0.2436 ± 0.0030 | 0.3862 ± 0.0111 |
| $\lambda = 10^0$ | 0.0 | 0.1564 ± 0.0741 | 0.2768 ± 0.0034 | 0.2426 ± 0.0032 | 0.4007 ± 0.0114 |
| $\lambda = 10^0$ | 1.0 | 0.2138 ± 0.0760 | 0.2794 ± 0.0033 | 0.2442 ± 0.0030 | 0.3839 ± 0.0112 |
| $\lambda = 10^1$ | 0.0 | 0.2701 ± 0.0711 | 0.2775 ± 0.0035 | 0.2470 ± 0.0031 | 0.3925 ± 0.0113 |
| $\lambda = 10^1$ | 1.0 | 0.3534 ± 0.0753 | 0.2809 ± 0.0034 | 0.2479 ± 0.0030 | 0.3776 ± 0.0112 |
| $\lambda = 10^{3/2}$ | 0.0 | 0.4662 ± 0.0722 | 0.2796 ± 0.0035 | 0.2539 ± 0.0030 | 0.3741 ± 0.0112 |
| $\lambda = 10^{3/2}$ | 1.0 | 0.5387 ± 0.0720 | 0.2827 ± 0.0034 | 0.2541 ± 0.0030 | 0.3625 ± 0.0108 |
| $\lambda = 10^2$ | 0.0 | 0.6949 ± 0.0721 | 0.2798 ± 0.0034 | 0.2616 ± 0.0030 | 0.3455 ± 0.0106 |
| $\lambda = 10^2$ | 1.0 | 0.7340 ± 0.0718 | 0.2827 ± 0.0034 | 0.2619 ± 0.0029 | 0.3319 ± 0.0100 |

*Metrics explanation:*
- **ImageReward**: Predicts human preference for image quality and alignment with prompt
- **CLIP-Score**: Measures semantic similarity between generated images and text prompts
- **HPS**: Human Preference Score - evaluates overall image aesthetics  
- **DreamSim Var**: Diversity metric based on DreamSim embeddings - higher values indicate more diverse outputs

## Clarifications

### Memoryless SOC schedules for DDIM
Following the [Adjoint Matching paper](https://openreview.net/forum?id=xQBRrtQM8u), reward fine-tuning diffusion and flow models requires using memoryless noise schedules. The [DDIM family](https://openreview.net/forum?id=St1giarCHLP) of schedules is parameterized by a one-dimensional parameter $\eta$. Any DDIM schedule with $\eta >= 1$ is memoryless by Prop. 1 of the Adjoint Matching paper, and setting $\eta = 1$ yields the distinguished memoryless schedule in Thm. 1. The config files use $\eta = 1$.

### Classifier-free guidance
Existing papers on SOC reward fine-tuning do not clarify how to adapt the methods to work with [classifier-free guidance (CFG)](https://openreview.net/pdf?id=qw8AKxfYbI). Empirically, CFG is critical to obtain high quality samples. There are different ways to incorporate CFG into reward fine-tuning, and this repo allows to experiment with them. The config files contain three arguments related to CFG:
- `guidance_scale`: the classifier-free guidance parameter, 1.0 for no guidance. The default in the config files is 7.5.
- `guidance_in_adjoint_computation`: if True, the base drift used in the adjoint computation is the CFG drift given by the base model, as opposed to the drift without guidance. The default in the config files is False. 
- `guidance_in_control_computation`: if True, the control in the loss function is constructed using the difference between the CFG drifts of the fine-tuned and base models, as opposed to the the difference between drifts without guidance. The default in the config files is False.

There are at least three different ways to handle CFG when fine-tuning, with different empirical performance and theoretical grounding:
- Fine-tuning with CFG trajectories (`guidance_scale=7.5, guidance_in_adjoint_computation=False, guidance_in_control_computation=False`): CFG used to sample fine-tuning trajectories, but not in the adjoint computation or the control computation. Not completely theoretically grounded, due to the mismatch between the generative SDE and the loss function. Great empirical performance.
- Fine-tuning without guidance (`guidance_scale=1.0, guidance_in_adjoint_computation=False, guidance_in_control_computation=False`): CFG used only at inference time. Theoretically grounded. Moderate empirical performance, because the fine-tuning sample are not high-quality since they do not use CFG.
- Full CFG fine-tuning (`guidance_scale=7.5, guidance_in_adjoint_computation=True, guidance_in_control_computation=True`): CFG used to sample fine-tuning trajectories, in the adjoint computation, and in the control computation. Theoretically grounded: the target distribution is the tilted version of the CFG distribution. Poor empirical performance, because the lean adjoint ODE solutions blow up in norm.

## Citation

If you use this code in your research, please consider citing the paper:

```bibtex
@inproceedings{
domingo-enrich2025adjoint,
title={Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control},
author={Carles Domingo-Enrich and Michal Drozdzal and Brian Karrer and Ricky T. Q. Chen},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=xQBRrtQM8u}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
