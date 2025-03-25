import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from far.utils.registry import DATASET_REGISTRY


def random_sample_frames(total_frames, num_frames, interval, split='training'):
    max_start = total_frames - (num_frames - 1) * interval

    if split == 'training':
        if max_start < 1:
            raise ValueError(f'Cannot sample {num_frames} from {total_frames} with interval {interval}')
        else:
            start = random.randint(0, max_start - 1)
    else:
        start = 0
        interval = 1 if max_start < 1 else interval

    frame_ids = [start + i * interval for i in range(num_frames)]

    return frame_ids


@DATASET_REGISTRY.register()
class DMLabDataset(Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.split = opt['split']

        self.data_cfg = opt['data_cfg']

        self.num_frames = self.data_cfg['num_frames']
        self.frame_interval = self.data_cfg['frame_interval']

        self.use_latent = opt.get('use_latent', False)

        with open(self.opt['data_list'], 'r') as fr:
            self.data_list = json.load(fr)

    def __len__(self):
        if self.opt.get('num_sample'):
            return self.opt['num_sample']
        else:
            return len(self.data_list)

    def read_video(self, video_path):
        data = np.load(video_path)
        total_frames = len(data['video'])

        frame_idxs = random_sample_frames(total_frames, self.num_frames, self.frame_interval, split=self.split)

        video = data['video'][frame_idxs]
        actions = data['actions'][frame_idxs]

        return video, actions

    def read_latent(self, latent_path, action_path=None):
        frames = torch.load(latent_path)
        total_frames = frames.shape[0]

        frame_idxs = random_sample_frames(total_frames, self.num_frames, self.frame_interval, split=self.split)

        frames = frames[frame_idxs]

        if action_path is not None:
            actions = np.load(action_path)['actions']
            actions = torch.from_numpy(actions[frame_idxs])
        else:
            actions = None

        return frames, actions

    def __getitem__(self, idx):
        if self.use_latent:
            latent_path, action_path = self.data_list[idx]['latent_path'], self.data_list[idx]['action_path']
            latent, actions = self.read_latent(latent_path, action_path=action_path)
            return {'latent': latent, 'action': actions, 'index': idx}
        else:
            video_path = self.data_list[idx]['video_path']
            video, actions = self.read_video(video_path)

            video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()
            actions = torch.from_numpy(actions)

            return {'video': video, 'action': actions, 'index': idx}
