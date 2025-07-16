from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from cotracker_core.cotracker_discriminator import TrajectoryDiscriminator
from omegaconf import OmegaConf
from einops import rearrange

class VideReward(nn.Module):
    def __init__(self, config_dict):
        """
        Args:
            config_dict: the config_dict for the traj_discriminator
        """
        super().__init__()
        arch_config = config_dict['discriminator_config']
        # initialize the traj_discriminator
        arch_config['train_updateformer'] = True #the custom updateformer is different from the one in the cotracker
        self.traj_discriminator = TrajectoryDiscriminator(**arch_config)

        # load the pretrained checkpoint
        pretrained_ckpt = config_dict['pretrained_ckpt']
        if pretrained_ckpt is not None:
            self.traj_discriminator = self.load_pretrained_discriminator(self.traj_discriminator, pretrained_ckpt)
        
        # freeze all the parameters of the traj_discriminator
        self.freeze_params()

        self.target_frame_size = arch_config['frame_size']
    
    def forward(self, vid):
        """
        Args:
            vid: the video to be evaluated, shape: (B, T, C, H, W), with pixel value range [0, 1]
        Returns:
            r: the reward, shape: (B,)
        """
        # resize the video by bilinear interpolation
        height, width = self.target_frame_size
        B, T = vid.shape[0], vid.shape[1]
        vid = rearrange(vid, 'b t c h w -> (b t) c h w')
        vid = F.interpolate(vid, size=(height, width), mode='bilinear', align_corners=False)
        vid = rearrange(vid, '(b t) c h w -> b t c h w', b=B, t=T)
        
        out = self.traj_discriminator.cotracker_forward(vid)
        r = F.sigmoid(out).squeeze(-1)
        
        return r

    def load_pretrained_discriminator(self, model, pretrained_ckpt):
        pretrained_ckpt = torch.load(pretrained_ckpt, map_location='cpu')
        pretrained_ckpt = pretrained_ckpt['discriminator_state_dict']
        # optimizer_state_dict = pretrained_ckpt['optimizer_state_dict']
        # global_step = pretrained_ckpt['global_step']
        # lr_scheduler_state_dict = pretrained_ckpt['lr_scheduler_state_dict']
        # latest_losses = pretrained_ckpt['latest_losses']

        msg = model.load_state_dict(pretrained_ckpt, strict=True)
        print('Loading pretrained checkpoint: ', msg)

        return model
    
    def freeze_params(self):
        self.traj_discriminator.eval()
        for param in self.traj_discriminator.parameters():
            param.requires_grad = False
    
    def unfreeze_updataformer(self):
        self.traj_discriminator.cotracker.model.updateformer.train()
        for name, param in self.traj_discriminator.named_parameters():
            if 'updateformer' in name:
                param.requires_grad = True
    
    def save_discriminator(self, path):
        # TO DO
        raise NotImplementedError("Saving the discriminator is not implemented yet")


def video_rm_load(
    traj_discriminator_config,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Load a ImageReward model

    Parameters
    ----------
    pretrained_ckpt : str
        The path to the pretrained checkpoint
    device : Union[str, torch.device]
        The device to put the loaded model
        
    Returns
    -------
    model : torch.nn.Module
        The Video reward model
    """
    traj_discriminator_config = OmegaConf.load(traj_discriminator_config)
    model = VideReward(traj_discriminator_config).to(device)

    model.eval()

    return model

if __name__ == "__main__":
    device = 'cuda:0'
    model = video_rm_load(traj_discriminator_config='/gpfs-flash/junlab/yexi24-postdoc/soc-fine-tuning-sd/configs/traj_discriminator.yaml', device=device)
    x = torch.randn(1, 16, 3, 576, 1024)
    x = torch.clip(x, 0, 1).to(device)
    
    out = model(x)
    print(out.shape)

    model.unfreeze_updataformer()