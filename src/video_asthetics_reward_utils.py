from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from einops import rearrange
from typing import Dict
from os.path import expanduser
import os
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
from PIL import Image
import open_clip


class VideoAestheticsReward(nn.Module):
    def __init__(self, config_dict):
        """
        Args:
            config_dict: the config_dict for the traj_discriminator
        """
        super().__init__()
        arch_config = config_dict['aesthetic_model_config']
        self.asthetics_model, self.clip_emb, self.normalize = self.load_aesthetic_model(clip_model=arch_config['clip_model'])
        
        # freeze all the parameters of the traj_discriminator
        self.freeze_params()

    def _forward(self, vid):
        """
        Args:
            vid: the video to be evaluated, shape: (B, T, C, H, W), with pixel value range [0, 1]
        Returns:
            r: the reward, shape: (B,)
        """
        # resize the video by bilinear interpolation
        height, width = 224, 224
        B, T = vid.shape[0], vid.shape[1]
        vid = rearrange(vid, 'b t c h w -> (b t) c h w')
        # if the original aspect ratio is not 1:1, resize the shorter side to 224, then do center crop
        # else, directly resize the video to 224x224
        h, w = vid.shape[2], vid.shape[3]
        if h != w:
            scale = height / min(h, w)
            new_h = max(height, int(h * scale + 0.5))
            new_w = max(width, int(w * scale + 0.5))
            vid = F.interpolate(vid, size=(new_h, new_w), mode='bilinear', align_corners=False)
            top = (new_h - height) // 2
            left = (new_w - width) // 2
            vid = vid[:, :, top:top + height, left:left + width]
        else:
            vid = F.interpolate(vid, size=(height, width), mode='bilinear', align_corners=False)

        vid = self.normalize(vid)
        image_features = self.clip_emb.encode_image(vid)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction = self.asthetics_model(image_features) #(b*t, 1)
        prediction = rearrange(prediction, '(b t) 1 -> b t', b=B, t=T)

        r = torch.mean(prediction, dim=1)

        return r

    def forward(self, vid):
        # forward with gradient checkpointing
        r = torch.utils.checkpoint.checkpoint(
            self._forward,
            vid,
            preserve_rng_state=False,
            use_reentrant=False
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return r
    
    def load_aesthetic_model(self, clip_model='ViT-L-14'):
        """load the aethetic model"""
        home = expanduser("~")
        cache_folder = home + "/.cache/emb_reader"
        if clip_model == 'ViT-L-14':
            clip_name = 'vit_l_14'
        elif clip_model == "ViT-B-32":
            clip_name = 'vit_b_32'
        else:
            raise ValueError()
        path_to_model = cache_folder + "/sa_0_4_"+clip_name+"_linear.pth"
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = (
                "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_name+"_linear.pth?raw=true"
            )
            urlretrieve(url_model, path_to_model)
        if clip_name == 'vit_l_14':
            m = nn.Linear(768, 1)
        elif clip_name == "vit_b_32":
            m = nn.Linear(512, 1)
        else:
            raise ValueError()
        s = torch.load(path_to_model)
        m.load_state_dict(s)
        m.eval()

        # load the clip model
        clip_emb, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained='openai')
        clip_emb.eval()

        # extract the transforms from preprocess
        steps = getattr(preprocess, "transforms", [preprocess])  # handle both Compose and single-callable
        normalize = steps[-1]
        
        return m, clip_emb, normalize

    def freeze_params(self):
        self.asthetics_model.eval()
        self.clip_emb.eval()
        for param in self.asthetics_model.parameters():
            param.requires_grad = False
        for param in self.clip_emb.parameters():
            param.requires_grad = False
    
    def unfreeze_updataformer(self):
        self.asthetics_model.train()
        self.clip_emb.train()
        for param in self.asthetics_model.parameters():
            param.requires_grad = True
        for param in self.clip_emb.parameters():
            param.requires_grad = True
    
    def save_asthetics_model(self, path):
        # TO DO
        raise NotImplementedError("Saving the asthetics model is not implemented yet")


def video_asthetics_rm_load(
    asthetics_model_config: Union[str, Dict],
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Load a ImageReward model

    Parameters
    ----------
    asthetics_model_config: str
        The config of asthetics model
    device : Union[str, torch.device]
        The device to put the loaded model
        
    Returns
    -------
    model : torch.nn.Module
        The Video reward model
    """
    if isinstance(asthetics_model_config, str):
        asthetics_model_config = OmegaConf.load(asthetics_model_config)
    model = VideoAestheticsReward(asthetics_model_config).to(device)

    model.eval()

    return model

if __name__ == "__main__":
    device = 'cuda:0'
    import os
    import torch
    import torch.nn as nn
    from os.path import expanduser  # pylint: disable=import-outside-toplevel
    from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
    from PIL import Image
    import open_clip

    def get_aesthetic_model(clip_model="vit_l_14"):
        """load the aethetic model"""
        home = expanduser("~")
        cache_folder = home + "/.cache/emb_reader"
        path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = (
                "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
            )
            urlretrieve(url_model, path_to_model)
        if clip_model == "vit_l_14":
            m = nn.Linear(768, 1)
        elif clip_model == "vit_b_32":
            m = nn.Linear(512, 1)
        else:
            raise ValueError()
        s = torch.load(path_to_model)
        m.load_state_dict(s)
        m.eval()
        return m

    amodel= get_aesthetic_model(clip_model="vit_l_14")
    amodel.eval()

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    pil_image = Image.open("lovely-cat-as-domestic-animal-view-pictures-182393057.jpg")

    import pdb; pdb.set_trace()
    from torchvision import transforms
    def apply_stepwise(compose, x):
        out = x
        steps = getattr(compose, "transforms", [compose])  # handle both Compose and single-callable
        for i, t in enumerate(steps):
            out = t(out)
            shape = getattr(out, "shape", getattr(out, "size", None))
            print(f"{i}: {t} -> {type(out).__name__}, {shape}")
        return out
    apply_stepwise(preprocess, pil_image)

    image = preprocess(pil_image).unsqueeze(0)
    with torch.no_grad():
        import pdb; pdb.set_trace()
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction = amodel(image_features)
        print(prediction)

    # model = video_rm_load(traj_discriminator_config='/gpfs-flash/junlab/yexi24-postdoc/soc-fine-tuning-sd/configs/traj_discriminator.yaml', device=device)
    # x = torch.randn(1, 16, 3, 576, 1024)
    # x = torch.clip(x, 0, 1).to(device)
    
    # out = model(x)
    # print(out.shape)

    # model.unfreeze_updataformer()