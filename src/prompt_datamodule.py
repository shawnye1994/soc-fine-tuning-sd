import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from core_utils import load_data

class PromptDataset(Dataset):
    def __init__(self, prompt_path="../prompt_files/refl_data.json"):
        self.prompts = load_data(prompt_path=prompt_path)
        if len(self.prompts) == 1:
            self.prompts = self.prompts * 2000 #*10000
        print(f"Loaded {len(self.prompts)} prompts from {prompt_path}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # Return only the prompt for on-policy sampling
        prompt = self.prompts[idx]
        return prompt


class PromptDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, training_prompt_path="../prompt_files/refl_data.json", validation_prompt_path="../prompt_files/benchmark_ir.json"):
        super().__init__()
        self.batch_size = batch_size
        self.training_prompt_path = training_prompt_path
        self.validation_prompt_path = validation_prompt_path
        self.setup()

    def setup(self, stage=None):
        self.train_dataset = PromptDataset(prompt_path=self.training_prompt_path)
        self.val_dataset = PromptDataset(prompt_path=self.validation_prompt_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=3,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=3,
        )