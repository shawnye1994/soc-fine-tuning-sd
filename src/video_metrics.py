import torch
from contextlib import nullcontext

from video_reward_utils import video_rm_load
# Stores the reward models
REWARDS_DICT = {
    "VideoReward": None,
}

# Returns the reward function based on the reward function name
def get_reward_function(reward_name, video):
    if reward_name == "VideoReward":
        return do_video_reward(video=video)

    else:
        raise ValueError(f"Unknown metric: {reward_name}")

# Compute VideoReward
def do_video_reward(*, videos, use_no_grad=True):
    global REWARDS_DICT
    assert REWARDS_DICT["VideoReward"] is not None, "VideoReward is not initialized"

    context = torch.no_grad() if use_no_grad else nullcontext()
    with context: 
        video_reward_result = REWARDS_DICT["VideoReward"](videos)

    return video_reward_result

# # Compute ImageReward
# def do_image_reward(*, images, prompts, use_no_grad=True, use_score_from_prompt_batched=False):
#     global REWARDS_DICT
#     if REWARDS_DICT["ImageReward"] is None:
#         REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0")

#     context = torch.no_grad() if use_no_grad else nullcontext()

#     with context:
#         if use_score_from_prompt_batched:
#             image_reward_result = REWARDS_DICT["ImageReward"].score_from_prompt_batched(prompts, images)
#         else:
#             image_reward_result = REWARDS_DICT["ImageReward"].score_batched(prompts, images)

#     return image_reward_result

def do_eval(*, videos, metrics_to_compute):
    """
    Compute the metrics for the given videos.
    """
    results = {}
    for metric in metrics_to_compute:
        if metric == "VideoReward":
            results[metric] = {}
            results[metric]["result"] = do_video_reward(videos=videos)

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results

def reward_function(video, traj_discriminator_config, device, reward_func="VideoReward", use_no_grad=False, verbose=False):
    """
    Computes reward values for generated videos using selected reward model.
    
    Args:
        video: Tensor of decoded videos to evaluate
        traj_discriminator_config: Configuration for the trajectory discriminator
        device: Device to run the computation on
        reward_func: Which reward function to use ("VideoReward")
        use_no_grad: Whether to disable gradient tracking
        
    Returns:
        Tensor of reward values (with or without gradients based on use_no_grad)
    """

    if reward_func == "VideoReward":
        global REWARDS_DICT
        if REWARDS_DICT["VideoReward"] is None:
            REWARDS_DICT['VideoReward'] = video_rm_load(traj_discriminator_config, device)

        rewards = do_video_reward(
            videos=video, 
            use_no_grad=use_no_grad
        )
    else:
        raise ValueError(f"Unknown metric: {reward_func}")

    if use_no_grad:
        return torch.tensor(rewards).to(video.device)
    else:
        if verbose:
            print(f'rewards.requires_grad in reward_function: {rewards.requires_grad}')
        return rewards
