import os
from typing import List

import imageio
import numpy as np
import torch
import wandb
from einops import rearrange
from PIL import Image


def log_paired_video(
    sample,
    gt=None,
    context_frames=0,
    save_suffix=None,
    save_dir=None,
    wandb_logger=None,
    wandb_cfg=None,
    annotate_context_frame=True
):

    # Add red border of 1 pixel width to the context frames
    if annotate_context_frame:
        color = [255, 0, 0]
        for i, c in enumerate(color):
            c = c / 255.0
            sample[:, :, :context_frames, i, [0, -1], :] = c
            sample[:, :, :context_frames, i, :, [0, -1]] = c
            if gt is not None:
                gt[:, :, :context_frames, i, [0, -1], :] = c
                gt[:, :, :context_frames, i, :, [0, -1]] = c
    if gt is not None:
        video = torch.cat([sample, gt], dim=-1).float().detach().cpu().numpy()
    else:
        video = sample.float().detach().cpu().numpy()
    video = (video.clip(0, 1) * 255).astype(np.uint8)
    video = rearrange(video, 'b n f c h w -> b (n f) h w c')

    os.makedirs(save_dir, exist_ok=True)

    for vid, idx in zip(video, save_suffix):

        save_video_to_dir(vid, save_dir=save_dir, save_suffix=f'sample_gt_{idx}', save_type='video', fps=8)

        if wandb_logger:
            vid = rearrange(vid, 'f h w c -> f c h w')
            wandb_logger.log({f"{wandb_cfg['namespace']}/sample_{idx}": wandb.Video(vid, fps=8)}, step=wandb_cfg['step'])


def save_video_to_dir(video, save_dir, save_suffix, save_type='frame', fps=8):
    if isinstance(video, np.ndarray):
        video = [Image.fromarray(frame).convert('RGB') for frame in video]
    elif isinstance(video, list):
        video = video
    else:
        raise NotImplementedError

    os.makedirs(save_dir, exist_ok=True)

    save_type_list = save_type.split('_')

    # save frame
    if 'frame' in save_type_list:
        frame_save_dir = os.path.join(save_dir, 'frames')
        os.makedirs(frame_save_dir, exist_ok=True)
        for idx, img in enumerate(video):
            img.save(os.path.join(frame_save_dir, f'{idx:05d}_{save_suffix}.jpg'))

    # save to gif
    if 'gif' in save_type_list:
        gif_save_path = os.path.join(save_dir, f'{save_suffix}.gif')
        save_images_as_gif(video, gif_save_path, fps=fps)

    # save to video
    if 'video' in save_type_list:
        video_save_path = os.path.join(save_dir, f'{save_suffix}.mp4')
        export_to_video(video, video_save_path, fps=fps)


def save_images_as_gif(images: List[Image.Image], save_path: str, fps=8) -> None:

    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=int(1000 / fps),
    )


def export_to_video(video_frames: List[Image.Image], output_video_path: str, fps=8) -> str:
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    video_writer = imageio.get_writer(output_video_path, fps=fps)

    # Write each image to the video
    for img in video_frames:
        video_writer.append_data(np.array(img))

    # Close the video writer
    video_writer.close()
    return output_video_path
