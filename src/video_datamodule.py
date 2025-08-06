import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from chunked_sampler import ChunkedSampler
import json
import torch
import os
import imageio.v3 as iio
import numpy as np
from torchvision import transforms


def read_video(vid_path, target_size, vid_data_type):
    """Read video file into a pytorch tensor, with shape (T, C, H, W), pixel range [0, 1]
    Args:
        vid_path: path to the video file
        target_size: the target size of the video, tuple of (T, H, W)
    Returns:
        video_tensor: pytorch tensor of shape (T, C, H, W), pixel range [0, 1]
        init_frame: PIL image of the first frame
    """
    if vid_data_type == 'mp4':
        # Read video file using imageio
        video = iio.imread(vid_path, index=None)  # Read all frames
        # Convert to numpy array if not already
        video_np = np.array(video)
        # Handle different video formats
        assert video_np.ndim == 4, "Video must have 4 dimensions (T, H, W, C)"

        # Convert from (T, H, W, C) to (T, C, H, W) format
        video_np = np.transpose(video_np, (0, 3, 1, 2))
        
        # Convert to PyTorch tensor
        video_tensor = torch.from_numpy(video_np).float()
        
        # Normalize to [0, 1]
        video_tensor = video_tensor / 255.0

    elif vid_data_type == 'pt':
        data = torch.load(vid_path)
        video_tensor = data['real_video'] #（T, C, H, W)
        assert video_tensor.min() >= 0 and video_tensor.max() <= 1, "Video tensor must be normalized to [0, 1]"

    # Take only the first target_size[0] frames
    assert video_tensor.shape[0] >= target_size[0], "Video must have at least target_size[0] frames"
    video_tensor = video_tensor[:target_size[0]]

    # Resize and center crop to target height and width
    target_h, target_w = target_size[1], target_size[2]
    
    # if the spatio ratio of target_h/target_w does not match the original spatio ratio of the video, then do center crop to the target size
    if video_tensor.shape[2] / float(video_tensor.shape[3]) != target_h / float(target_w):
        # Create transform pipeline for resizing and cropping
        resize_transform = transforms.Compose([
            transforms.Resize(max(target_h, target_w)),  # Resize the smaller dimension
            transforms.CenterCrop((target_h, target_w))  # Center crop to target size
        ])
    else:
        resize_transform = transforms.Resize((target_h, target_w))
    
    # Apply transforms to each frame
    resized_frames = []
    for i in range(video_tensor.shape[0]):
        frame = video_tensor[i]
        resized_frame = resize_transform(frame)
        resized_frames.append(resized_frame.unsqueeze(0))
    
    # Stack frames back into a video tensor
    video_tensor = torch.cat(resized_frames, dim=0)
    init_frame = video_tensor[0, ...]

    return video_tensor, init_frame

class VideoDataset(Dataset):
    def __init__(self, video_path_json="refl_videos.json", target_vid_size=(24, 576, 1024), vid_data_type='mp4'):
        """
        Args:
            video_path_json: path to the json file containing the video paths
            target_vid_size: the target size of the video, tuple of (num_frames, height, width)
        """
        with open(video_path_json, "r") as f:
            data = json.load(f)
        self.data = data
        self.vid_ids = list(data.keys())
        self.target_vid_size = target_vid_size
        self.vid_data_type = vid_data_type

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, idx):
        """Return the reference real video for on-policy sampling
        Return:
            video_tensor: pytorch tensor of shape (T, C, H, W), pixel range [0, 1]
            init_frame: PIL image of the first frame
        """
        vid_path = self.data[self.vid_ids[idx]]
        video_tensor, init_frame = read_video(vid_path, self.target_vid_size, self.vid_data_type)

        return video_tensor, init_frame

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=3, train_video_path_json="refl_videos.json", val_video_path_json="refl_videos.json", target_vid_size=(24, 576, 1024), vid_data_type='mp4'):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_video_path_json = train_video_path_json
        self.val_video_path_json = val_video_path_json
        self.target_vid_size = target_vid_size
        self.vid_data_type = vid_data_type
        self.setup()

    def setup(self, stage=None):
        self.train_dataset = VideoDataset(video_path_json=self.train_video_path_json, target_vid_size=self.target_vid_size, vid_data_type=self.vid_data_type)
        self.val_dataset = VideoDataset(video_path_json=self.val_video_path_json, target_vid_size=self.target_vid_size, vid_data_type=self.vid_data_type)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=3,
        )

class BufferVideoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, buffer_size, num_workers=3, train_video_path_json="refl_videos.json", 
                 val_video_path_json="refl_videos.json", target_vid_size=(24, 576, 1024),
                 vid_data_type='mp4'):
        super().__init__()
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.train_video_path_json = train_video_path_json
        self.val_video_path_json = val_video_path_json
        self.target_vid_size = target_vid_size
        self.vid_data_type = vid_data_type
        self.sampler = None
        self.setup()

    def setup(self, stage=None):
        """
        Called by Lightning before training (or testing).
        """
        self.train_dataset = VideoDataset(video_path_json=self.train_video_path_json, target_vid_size=self.target_vid_size, vid_data_type=self.vid_data_type)
        self.val_dataset = VideoDataset(video_path_json=self.val_video_path_json, target_vid_size=self.target_vid_size, vid_data_type=self.vid_data_type)

    def train_dataloader(self):
        self.sampler = ChunkedSampler(self.train_dataset, chunk_size=self.buffer_size, shuffle=True)
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            drop_last=True  # ensures consistent batch size if you want
        )
        return dataloader
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=3,
        )
    
if __name__ == '__main__':
    video_dataset = VideoDataset('/gpfs-flash/junlab/yexi24-postdoc/refl_videos.json')
    print(len(video_dataset))

    import pdb; pdb.set_trace()
    a = [video_dataset[i] for i in [0,1,2]]
    
    vid_tensor, init_frame = video_dataset.__getitem__(1)
    import pdb; pdb.set_trace()