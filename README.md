# Adjoint Matching for Stable Diffusion

## Overview
This repository provides an implementation of Adjoint Matching (AM) for Stable Diffusion 1.5. It adds specialized trainers, custom schedulers, and prompt-based dataloaders.

## Features
- Adjoint Matching trainer class (AMTrainer), which can be repurposed for other SOC fine-tuning algorithms
- Custom DDIM scheduler (CustomDDIMScheduler)
- PromptDataModule for training and evaluation prompts
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
```
python src/train.py --config configs/multi_prompt.yaml
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
