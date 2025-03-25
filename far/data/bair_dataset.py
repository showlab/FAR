import json
import random

import decord
import torch
from torch.utils.data import Dataset

from far.utils.registry import DATASET_REGISTRY

decord.bridge.set_bridge('torch')


def random_sample_frames(total_frames, num_frames, interval, split='training'):
    max_start = total_frames - (num_frames - 1) * interval

    if split == 'training' and max_start > 0:
        start = random.randint(0, max_start - 1)
    else:
        start = 0
        interval = 1

    frame_ids = [start + i * interval for i in range(num_frames)]

    return frame_ids


@DATASET_REGISTRY.register()
class BairDataset(Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.split = opt['split']

        self.data_cfg = opt['data_cfg']

        self.n_frames = self.data_cfg['n_frames']
        self.frame_interval = self.data_cfg['frame_interval']

        with open(self.opt['data_list'], 'r') as fr:
            self.data_list = json.load(fr)

    def __len__(self):
        if self.opt.get('num_sample'):
            return self.opt['num_sample']
        else:
            return len(self.data_list)

    def read_video(self, video_path):
        video_reader = decord.VideoReader(video_path)
        total_frames = len(video_reader)

        if total_frames < self.n_frames:
            frame_idxs = list(range(total_frames))
            frames = video_reader.get_batch(frame_idxs)
            pad_frames = torch.zeros((self.n_frames - total_frames, *frames.shape[1:]))
            frames = torch.cat([frames, pad_frames], dim=0)
        else:
            frame_idxs = random_sample_frames(total_frames, self.n_frames, self.frame_interval, split=self.split)
            frames = video_reader.get_batch(frame_idxs)
        return frames

    def __getitem__(self, idx):
        video_path, _ = self.data_list[idx]['video_path'], self.data_list[idx]['action_path']

        video = self.read_video(video_path)
        video = (video / 255.0).float().permute(0, 3, 1, 2).contiguous()

        return {'video': video, 'index': idx}
