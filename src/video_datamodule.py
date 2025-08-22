import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from chunked_sampler import ChunkedSampler
import json
import torch
import os
import imageio.v3 as iio
import numpy as np
from torchvision import transforms
import glob
import random
from collections import defaultdict


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
        scale_h, scale_w = float(target_h) / video_tensor.shape[2], float(target_w) / video_tensor.shape[3]
        scale_factor = max(scale_h, scale_w)
        # Calculate intermediate size
        intermediate_h = int(video_tensor.shape[2] * scale_factor)
        intermediate_w = int(video_tensor.shape[3] * scale_factor)
        # Create transform pipeline
        resize_transform = transforms.Compose([
            transforms.Resize((intermediate_h, intermediate_w)),
            transforms.CenterCrop((target_h, target_w))
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
        if len(self.train_dataset) % self.buffer_size != 0:
            # round the length of train_dataset to be the largest multiple of buffer_size by split
            new_length = len(self.train_dataset) - len(self.train_dataset) % self.buffer_size
            # Modify the vid_ids list instead of slicing the dataset
            self.train_dataset.vid_ids = self.train_dataset.vid_ids[:new_length]

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

def create_video_json(directory, output_file="video_files.json", recursive=False):
    """
    List all MP4 video files in a directory and save their absolute paths to a JSON file.
    
    Args:
        directory (str): Path to the directory to scan for MP4 files
        output_file (str): Path to the output JSON file (default: "video_files.json")
        recursive (bool): Whether to search subdirectories recursively (default: False)
    
    Returns:
        dict: Dictionary mapping string indices to absolute file paths
    """
    # Check if directory exists
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    
    if not os.path.isdir(directory):
        raise ValueError(f"Path is not a directory: {directory}")
    
    # Get absolute path of the directory
    abs_directory = os.path.abspath(directory)
    
    # Find all MP4 files in the directory
    if recursive:
        mp4_pattern = os.path.join(abs_directory, "**", "*.mp4")
        mp4_files = glob.glob(mp4_pattern, recursive=True)
    else:
        mp4_pattern = os.path.join(abs_directory, "*.mp4")
        mp4_files = glob.glob(mp4_pattern)
    
    # Sort files for consistent ordering
    mp4_files.sort()
    
    # Create dictionary with string indices as keys
    video_dict = {}
    for i, file_path in enumerate(mp4_files):
        video_dict[str(i)] = os.path.abspath(file_path)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(video_dict, f, indent=4)
    
    print(f"Found {len(mp4_files)} MP4 files in {abs_directory}")
    if recursive:
        print("(searched recursively)")
    print(f"Saved to {output_file}")
    
    return video_dict

def split_videos_by_object(input_json_path, validation_ratio=0.2, seed=42, output_dir=None):
    """
    Split videos by object into training and validation sets.
    
    Args:
        input_json_path (str): Path to the input JSON file containing video paths
        validation_ratio (float): Ratio of videos to use for validation (0.0 to 1.0)
        seed (int): Random seed for reproducible splits
        output_dir (str): Directory to save output files (default: same as input)
    
    Returns:
        tuple: (train_dict, val_dict) - dictionaries for training and validation sets
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read the input JSON file
    with open(input_json_path, 'r') as f:
        video_dict = json.load(f)
    
    # Group videos by object name
    object_videos = defaultdict(list)
    
    for key, video_path in video_dict.items():
        # Extract object name from filename
        filename = os.path.basename(video_path)
        # Remove file extension and sequence number to get object name
        # Example: "bear_s00000.mp4" -> "bear"
        object_name = filename.split('_s')[0] if '_s' in filename else filename.split('.')[0]
        object_videos[object_name].append((key, video_path))
    
    print(f"Found {len(object_videos)} different objects:")
    for obj_name, videos in object_videos.items():
        print(f"  {obj_name}: {len(videos)} videos")
    
    # Split each object's videos into train/val
    train_videos = {}
    val_videos = {}
    train_counter = 0
    val_counter = 0
    
    for object_name, videos in object_videos.items():
        # Shuffle videos for this object
        random.shuffle(videos)
        
        # Calculate split point
        num_videos = len(videos)
        num_val = max(1, int(num_videos * validation_ratio))  # At least 1 video for validation
        num_train = num_videos - num_val
        
        print(f"\n{object_name}: {num_train} train, {num_val} validation")
        
        # Split videos
        train_object_videos = videos[:num_train]
        val_object_videos = videos[num_train:]
        
        # Add to train set with new indices
        for original_key, video_path in train_object_videos:
            train_videos[str(train_counter)] = video_path
            train_counter += 1
        
        # Add to validation set with new indices
        for original_key, video_path in val_object_videos:
            val_videos[str(val_counter)] = video_path
            val_counter += 1
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_json_path)
    
    # Save training set
    train_output_path = os.path.join(output_dir, "train_videos.json")
    with open(train_output_path, 'w') as f:
        json.dump(train_videos, f, indent=4)
    
    # Save validation set
    val_output_path = os.path.join(output_dir, "val_videos.json")
    with open(val_output_path, 'w') as f:
        json.dump(val_videos, f, indent=4)
    
    print(f"\nSaved {len(train_videos)} training videos to {train_output_path}")
    print(f"Saved {len(val_videos)} validation videos to {val_output_path}")
    
    return train_videos, val_videos


if __name__ == '__main__':
    # video_dataset = VideoDataset('/gpfs-flash/junlab/yexi24-postdoc/refl_videos.json')
    # print(len(video_dataset))

    # import pdb; pdb.set_trace()
    # a = [video_dataset[i] for i in [0,1,2]]
    
    # vid_tensor, init_frame = video_dataset.__getitem__(1)
    # import pdb; pdb.set_trace()


    # directory_path = "/gpfs/junlab/yexi24-postdoc/wenjia/datasets/davis_2017/real_videos_seq"
    # result = create_video_json(directory_path, "/gpfs-flash/junlab/yexi24-postdoc/soc-fine-tuning-sd/configs/refl_davis_videos.json")
    # print(f"Created JSON with {len(result)} videos")

    # input_file = "/gpfs-flash/junlab/yexi24-postdoc/soc-fine-tuning-sd/configs/refl_davis_videos.json"
    # output_dir = "/gpfs-flash/junlab/yexi24-postdoc/soc-fine-tuning-sd/configs"
    # # Split with 20% validation ratio
    # train_dict, val_dict = split_videos_by_object(
    #     input_json_path=input_file,
    #     validation_ratio=0.1,
    #     seed=42,
    #     output_dir=output_dir
    # )
    
    # print(f"\nFinal split:")
    # print(f"Training set: {len(train_dict)} videos")
    # print(f"Validation set: {len(val_dict)} videos")

    vid_path = '/gpfs/junlab/yexi24-postdoc/wenjia/datasets/davis_2017/real_videos_seq/bear_s00000.mp4'
    target_size = (24, 320, 576)
    vid_data_type = 'mp4'
    video_tensor, init_frame = read_video(vid_path, target_size, vid_data_type)
    # save the init_frame tensor as pil image
    init_frame = transforms.ToPILImage()(init_frame)
    init_frame.save('init_frame_new.png')

    
