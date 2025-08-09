import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from .cotracker_source.predictor import CoTrackerPredictor
class TrajectoryDiscriminator(nn.Module):
    def __init__(self, frame_size = [256, 256], pretrain_cotracker_ckpt_file='/gpfs/junlab/yexi24-postdoc/cotracker_checkpoints/scaled_offline.pth',
                 spatio_query_method='Grid', time_query_method='First', train_updateformer=False,
                 max_num_queries=1024):#, feature_preprocess_method='bilinear_scatter'):
        super(TrajectoryDiscriminator, self).__init__()
        self.frame_size = frame_size

        self.spatio_query_method = spatio_query_method
        self.time_query_method = time_query_method
        self.max_num_queries = max_num_queries

        # init the cotracker
        self.cotracker = CoTrackerPredictor(
                checkpoint=pretrain_cotracker_ckpt_file,
                v2=False,
                offline=True,
                window_len=60,
                train_updateformer=train_updateformer
            )
        
        # freeze all parameters of the cotracker
        self.cotracker.eval()
        for p in self.cotracker.parameters():
            p.requires_grad_(False)
        if train_updateformer:
            # enable the training of the updateformer
            self.cotracker.model.updateformer.train()
            for p in self.cotracker.model.updateformer.parameters():
                p.requires_grad_(True)

    def forward(self, forward_mode, vid, real=True):
        if forward_mode == 'g_step':
            cls_logits = self.forward_g_step(vid)
            return cls_logits
        elif forward_mode == 'd_step':
            cls_logits = self.forward_d_step(vid, real)
            return cls_logits
        elif forward_mode == 'pretrain':
            cls_logits = self.forward_pretrain(vid)
            return cls_logits
        else:
           raise ValueError(f"Unsupported forward mode: {forward_mode}")
        
    def forward_g_step(self, vid):
        """Forward pass for GAN generator update step
        Args:
            vid: input video tensor with shape of (B T C H W)
        Returns:
            cls_logits: (B, 1), the logits for the discriminator
        """
        cls_logits = self.cotracker_forward(vid)
        return cls_logits
        # return cls_logits, vid
    
    def forward_d_step(self, vid, real=True):
        if real:
            # for later R1 gradient penalty
            vid.requires_grad_(True)
            vid.retain_grad()
        
        cls_logits = self.cotracker_forward(vid)
        return cls_logits
    
    def forward_pretrain(self, vid):
        cls_logits = self.cotracker_forward(vid)

        return cls_logits
    
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def cotracker_forward(self, x):
        """
        Args:
            x: input video tensor with shape of (B T C H W), pixel value range: [0,1]
        Returns:
            cls_logits: the logits for the discriminator, tensor with shape of (B, 1)
        """
        # assert that x is in range of [0, 1]
        assert x.min() >= 0 and x.max() <= 1, f"input video for cotracker is not in range of [0, 1], min: {x.min()}, max: {x.max()}"
        # convert x to range of [0, 255]
        x = x * 255
        # generate the queries
        queries = self.__queries_generate__(x)
        queries = queries.to(x.dtype)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            cls_logits = self.cotracker(x, queries=queries) # B T N 2,  B T N 1

        return cls_logits
    
    def __queries_generate__(self, vid):
        """Generate the qureies for the cotracker
        # the queries coordinates (for spatial_query_method == 'Random') are also the index of queries point embedding
        Args:
            vid: input video tensor with shape of (B T C H W), pixel value range: [0,255]
        Returns:
            queries: the queries for the cotracker, tensor with shape of (B N 3), where N is the number of query points
            Last dim of queries is [t, x, y], where x is "W-dim and y is H-dim"!!!!
        """
        # generate the query time
        B, T, C, H, W = vid.shape
        if self.spatio_query_method == 'Grid':
            # official spatial grid query method of CoTracker, Not suitable for the training of track discriminator
            # query_s = get_points_on_a_grid(size=int(self.max_num_queries**0.5), extent=(H, W), device=vid.device)
            # modified from https://github.com/facebookresearch/co-tracker/blob/main/cotracker/models/core/model_utils.py#L83
            extent = (H, W)
            center = [extent[0] / 2, extent[1] / 2]
            margin = W / 64
            range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
            range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(*range_y, int(self.max_num_queries**0.5), device=vid.device),
                torch.linspace(*range_x, int(self.max_num_queries**0.5), device=vid.device),
                indexing="ij",
            )
            #note that here grid for W-dim/grid_x is placed before grid for H-dim/grid_y
            query_s = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
            query_s = repeat(query_s, 'h w c -> b (h w) c', b=B)

        elif self.spatio_query_method == 'Random':
            # randomly select max_num_queries points from the dense grid, which is the default setting for training
            extent = (H, W)
            range_y = (0, extent[0])
            range_x = (0, extent[1])
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, H, device=vid.device),
                torch.arange(0, W, device=vid.device),
            )
            query_s = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
            query_s = rearrange(query_s, 'h w c -> (h w) c')
            # randomly select max_num_queries points
            assert self.max_num_queries <= query_s.shape[0], "max_num_queries should be less than HxW"
            batch_query_s = []
            for i in range(B):
                indices = torch.randperm(query_s.shape[0])[:self.max_num_queries]
                batch_query_s.append(query_s[indices, :])
            query_s = torch.stack(batch_query_s, dim=0).to(torch.float32)
        else:
            raise NotImplementedError(f"Unsupported spatio query method: {self.spatio_query_method}")
        
        # generate the query spatial coordinates
        if self.time_query_method == 'First':
            query_t = torch.zeros(1, dtype=torch.long)
            query_t = repeat(query_t, '1 -> b n 1', b=B, n=query_s.shape[1])
        elif self.time_query_method == 'Last':
            query_t = torch.ones(1, dtype=torch.long)*(T-1)
            query_t = repeat(query_t, '1 -> b n 1', b=B, n=query_s.shape[1])
        elif self.time_query_method == 'Random':
            # randomly select B*query_s.shape[1] time points from the range (0, T)
            query_t = torch.randint(0, T, (B*query_s.shape[1],))
            query_t = rearrange(query_t, '(b n) -> b n 1', b=B, n=query_s.shape[1])
        else:
            raise NotImplementedError(f"Unsupported time query method: {self.time_query_method}")
        query_t = query_t.to(vid.device)

        #concatenate the time and spatial queries
        queries = torch.cat([query_t, query_s], dim=-1)

        return queries

if __name__ == '__main__':
    from accelerate import Accelerator
    import wandb
    import numpy as np
    from tqdm.auto import tqdm
    import argparse

    device = 'cuda:5'
    
    track_disc = TrajectoryDiscriminator().to(device)

    #print total number of parameters
    print('total parameters:', sum(p.numel() for p in track_disc.parameters()))
    #print number of trainable parameters
    print('trainable parameters:', sum(p.numel() for p in track_disc.parameters() if p.requires_grad))
    # generate random video with in range [0, 255]
    video = torch.randint(0, 256, (1, 16, 3, 256, 256)).to(device)
    vid = video / 255.0

    # vid = torch.randn(1, 16, 3, 256, 256).to(device)
    vid.requires_grad_(True)

    cls_logits, x = track_disc.forward_g_step(vid)
    loss = -cls_logits.mean()

    loss.backward()
    print(vid.grad.shape)
    print(x.grad.shape)

    temp = x.grad.mean(dim=2, keepdim=True)
    diff = (vid.grad[:, :, 2:3, ...] - temp).abs()
    print(diff.mean(), diff.sum(), diff.max(), diff.min())
    
    # vid = vid.detach()
    # cls_logits, real_track_feature = track_disc.forward_d_step(vid, real=True)
    # loss = -cls_logits.mean()
    # loss.backward()
    # print(real_track_feature.grad.shape)



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 accelerate launch --num_processes 10 --main_process_port 29513 track_discriminator.py --epochs 100 --batch_size 20
# CUDA_VISIBLE_DEVICES=5,6,7,8,9 accelerate launch --num_processes 5 --main_process_port 29513 track_discriminator.py --epochs 100 --batch_size 80
