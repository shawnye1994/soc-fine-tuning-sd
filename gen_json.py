import json

def create_video_config_json(video_path, n_entries, output_file):
    """
    Creates a JSON configuration file with numbered IDs from 0 to n_entries-1,
    all pointing to the same video path.
    
    Args:
        video_path (str): Path to the video file
        n_entries (int): Number of entries to create (creates IDs 0 to n_entries-1)
        output_file (str): Path where to save the JSON file
    """
    config = {}
    for i in range(n_entries):
        config[str(i)] = video_path
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Created {output_file} with {n_entries} entries pointing to {video_path}")

# Example usage:
if __name__ == "__main__":
    # Recreate your example file
    video_path = "/gpfs/junlab/yexi24-postdoc/wenjia/datasets/davis_2017/real_videos_seq/bear_s00044.mp4"
    create_video_config_json(video_path, 8, "val_davis_debug.json")