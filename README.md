# Adjoint Matching for Stable Diffusion

## Overview
This repository provides an implementation of reward fine-tuning methods for Stable Diffusion 1.5 based on stochastic optimal control (SOC), including [Adjoint Matching](https://openreview.net/forum?id=xQBRrtQM8u). It adds specialized trainers, custom schedulers, and prompt-based dataloaders.

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
git clone https://github.com/your-username/adjoint-matching-SD.git
cd adjoint-matching-SD
conda create -n AM-env python=3.10.16
conda activate AM-env
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
