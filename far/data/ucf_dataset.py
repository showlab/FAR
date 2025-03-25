import json
import math
import random

import decord
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from far.utils.registry import DATASET_REGISTRY

decord.bridge.set_bridge('torch')


def random_sample_frames(total_frames, num_frames, interval, split='training'):
    max_start = total_frames - (num_frames - 1) * interval

    if split == 'training':
        if max_start < 1:
            start = 0
            interval = 1
        else:
            start = random.randint(0, max_start - 1)
    else:
        start = 0
        if max_start < 1:
            interval = 1

    frame_ids = [start + i * interval for i in range(num_frames)]

    return frame_ids


def get_balanced_dataset(data_list, num_samples):  # for ucf-101 fvd evaluation
    data_by_class = {}
    for item in data_list:
        label = item['label']
        if label not in data_by_class:
            data_by_class[label] = []
        data_by_class[label].append(item)

    num_classes = len(data_by_class)
    samples_per_class = math.ceil(num_samples / num_classes)

    balanced_data_list = []

    for label, items in data_by_class.items():
        sampled_items = random.sample(items, samples_per_class)
        balanced_data_list.extend(sampled_items)

    random.shuffle(balanced_data_list)
    return balanced_data_list[:num_samples]


@DATASET_REGISTRY.register()
class UCFDataset(Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.split = opt['split']

        self.data_cfg = opt['data_cfg']

        self.n_frames = self.data_cfg['n_frames']
        self.frame_interval = self.data_cfg['frame_interval']

        self.use_latent = opt.get('use_latent', False)

        with open(self.opt['data_list'], 'r') as fr:
            self.data_list = json.load(fr)

        if self.split == 'training':
            self.transform = transforms.Compose([
                transforms.Resize(self.data_cfg['resolution']),
                transforms.RandomCrop(self.data_cfg['resolution'])
            ])
        else:
            if self.data_cfg['evaluation_type'] == 'MCVD':
                self.data_list = self.data_list[::10]
            elif self.data_cfg['evaluation_type'] == 'Latte':
                self.data_list = get_balanced_dataset(self.data_list, self.opt['num_sample'])
            else:
                raise NotImplementedError

            self.transform = transforms.Compose([
                transforms.Resize(self.data_cfg['resolution']),
                transforms.CenterCrop(self.data_cfg['resolution'])
            ])

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

    def read_latent(self, latent_path):
        frames = torch.load(latent_path)
        total_frames = frames.shape[0]
        if total_frames < self.n_frames:
            frame_idxs = list(range(total_frames))
            frames = frames[frame_idxs]
            pad_frames = torch.zeros((self.n_frames - total_frames, *frames.shape[1:]))
            frames = torch.cat([frames, pad_frames], dim=0)
        else:
            frame_idxs = random_sample_frames(total_frames, self.n_frames, self.frame_interval, split=self.split)
            frames = frames[frame_idxs]
        return frames

    def __getitem__(self, idx):
        if self.use_latent:
            raise NotImplementedError
        else:
            video_path, label = self.data_list[idx]['video_path'], self.data_list[idx]['label']
            video = self.read_video(video_path)

            video = (video / 255.0).float().permute(0, 3, 1, 2).contiguous()
            video = self.transform(video)

            if self.data_cfg.get('use_flip') and random.random() < 0.5:
                video = torch.flip(video, dims=(3, ))

            return {'video': video, 'label': label, 'index': idx}
