# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .core.cotracker.cotracker import CoTracker2
from .core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from .core.cotracker.cotracker3_online import CoTrackerThreeOnline


def build_cotracker(
    checkpoint: str,
):
    if checkpoint is None:
        return build_cotracker()
    model_name = checkpoint.split("/")[-1].split(".")[0]
    if model_name == "cotracker":
        return build_cotracker(checkpoint=checkpoint)
    else:
        raise ValueError(f"Unknown model name {model_name}")


def build_cotracker(checkpoint=None, offline=True, window_len=16, v2=False, train_updateformer=False):
    if offline:
        cotracker = CoTrackerThreeOffline(
            stride=4, corr_radius=3, window_len=window_len
        )
    else:
        if v2:
            cotracker = CoTracker2(stride=4, window_len=window_len)
        else:
            cotracker = CoTrackerThreeOnline(
                stride=4, corr_radius=3, window_len=window_len
            )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]

        strict = True
        if train_updateformer:
            strict = False
            # Filter out the keys that start with 'updateformer'
            # We need to train the updateformer from scratch
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('updateformer')}
            # also remove time_emb from the state_dict
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('time_emb')}

        cotracker.load_state_dict(state_dict, strict=strict)
    return cotracker
