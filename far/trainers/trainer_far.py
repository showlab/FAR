import copy
import json
import os
import random
from glob import glob

import torch
from accelerate.logging import get_logger
from accelerate.utils import broadcast
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from einops import rearrange
from pytorchvideo.data.encoded_video import EncodedVideo
from safetensors.torch import load_file
from tqdm import tqdm

from far.metrics.metric import VideoMetric
from far.models import build_model
from far.models.autoencoder_dc_model import MyAutoencoderDC
from far.pipelines import build_pipeline
from far.utils.ema_util import EMAModel
from far.utils.registry import TRAINER_REGISTRY
from far.utils.vis_util import log_paired_video


@TRAINER_REGISTRY.register()
class FARTrainer:

    def __init__(
        self,
        accelerator,
        model_cfg,
        clean_context_ratio,
        weighting_scheme='uniform',
        context_timestep_idx=-1,
        training_type='base'
    ):
        super(FARTrainer, self).__init__()

        self.accelerator = accelerator
        weight_dtype = torch.float32
        if accelerator.mixed_precision == 'fp16':
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == 'bf16':
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        # build model
        if model_cfg['transformer']['from_pretrained']:
            raise NotImplementedError
        else:
            init_cfg = model_cfg['transformer']['init_cfg']
            self.model = build_model(
                init_cfg['type'])(**init_cfg.get('config', {}))
            if init_cfg.get('pretrained_path'):
                state_dict = torch.load(init_cfg['pretrained_path'], map_location='cpu', weights_only=True)
                self.model.load_state_dict(state_dict)

        if model_cfg['transformer'].get('gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()

        if model_cfg['vae'].get('from_pretrained'):
            self.vae = build_model(model_cfg['vae']['type']).from_pretrained(
                model_cfg['vae']['from_pretrained']).to(accelerator.device, dtype=weight_dtype)
            self.vae.requires_grad_(False)
        elif model_cfg['vae'].get('from_config'):
            with open(model_cfg['vae']['from_config'], 'r') as fr:
                config = json.load(fr)
            self.vae = build_model(model_cfg['vae']['type']).from_config(config)
            if model_cfg['vae'].get('from_config_pretrained'):
                state_dict_path = model_cfg['vae']['from_config_pretrained']
                if state_dict_path.endswith('.safetensors'):
                    state_dict = load_file(model_cfg['vae']['from_config_pretrained'])
                else:
                    state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
                self.vae.load_state_dict(state_dict)
            self.vae.to(accelerator.device, dtype=weight_dtype)
            self.vae.requires_grad_(False)
        else:
            raise NotImplementedError

        if model_cfg['scheduler']['from_pretrained']:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_cfg['scheduler']['from_pretrained'], subfolder='scheduler')
        else:
            raise NotImplementedError

        self.weighting_scheme = weighting_scheme
        self.ema = None
        self.clean_context_ratio = clean_context_ratio
        self.context_timestep_idx = context_timestep_idx
        self.training_type = training_type

    def set_ema_model(self, ema_decay):
        logger = get_logger('far', log_level='INFO')

        if ema_decay is not None:
            self.ema = EMAModel(self.accelerator.unwrap_model(self.model), decay=ema_decay)
            logger.info(f'enable EMA training with decay {ema_decay}')

    def get_params_to_optimize(self, param_names_to_optimize):
        logger = get_logger('far', log_level='INFO')

        params_to_optimize = []
        params_to_fix = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_optimize.append(param)
                logger.info(f'optimize params: {name}')
            else:
                params_to_fix.append(param)
                logger.info(f'fix params: {name}')

        logger.info(
            f'#Trained Parameters: {sum([p.numel() for p in params_to_optimize]) / 1e6} M'
        )
        logger.info(
            f'#Fixed Parameters: {sum([p.numel() for p in params_to_fix]) / 1e6} M'
        )

        return params_to_optimize

    def sample_frames(self, batch):
        video = batch['video'] if 'video' in batch else batch['latent']
        total_frames = video.shape[1]

        num_sample_frames = torch.randint(
            low=self.accelerator.unwrap_model(self.model).config.short_term_ctx_winsize,
            high=total_frames,
            size=(1, ),
            device=self.accelerator.device)
        num_sample_frames = broadcast(num_sample_frames)

        start_frame_idx = random.randint(0, total_frames - num_sample_frames)

        video = video[:, start_frame_idx:start_frame_idx + num_sample_frames]

        if 'label' in batch:
            raise NotImplementedError
        elif 'action' in batch:
            batch['action'] = batch['action'][:, start_frame_idx:start_frame_idx + num_sample_frames]
        else:
            raise NotImplementedError

        if 'video' in batch:
            batch['video'] = video
        else:
            batch['latent'] = video

        return batch

    def train_step(self, batch, iters=-1):
        self.vae.eval()
        self.model.train()

        if self.training_type == 'long_context':
            batch = self.sample_frames(batch)

        if 'video' in batch:
            video = batch['video'].to(dtype=self.weight_dtype)  # 0-1
            if isinstance(self.vae, MyAutoencoderDC):
                batch_size, num_frames = video.shape[:2]
                video = rearrange(video, 'b t c h w -> (b t) c h w')
                latents = self.vae.encode((video * 2 - 1)).latent
            else:
                raise NotImplementedError
        else:
            latents = batch['latent'].to(dtype=self.weight_dtype)
            batch_size, num_frames = latents.shape[0], latents.shape[1]
            latents = rearrange(latents, 'b t c h w -> (b t) c h w')

        if 'label' in batch:
            conditions = {'label': batch['label']}
        elif 'action' in batch:
            conditions = {'action': batch['action']}
        else:
            conditions = None

        latents = latents * self.vae.config.scaling_factor

        # Sample noise
        noise = torch.randn_like(latents)

        # Sample a random timestep for each image
        # flow matching requires float timesteps (retrieve from scheduler)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=batch_size * num_frames,
            logit_mean=0,
            logit_std=1,
        )

        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device=latents.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.scale_noise(latents, timesteps, noise)

        timesteps = rearrange(timesteps, '(b t) -> b t', b=batch_size)
        latents = rearrange(latents, '(b t) c h w -> b t c h w', b=batch_size)
        noisy_latents = rearrange(noisy_latents, '(b t) c h w -> b t c h w', b=batch_size)
        noise = rearrange(noise, '(b t) c h w -> b t c h w', b=batch_size)

        short_term_ctx_winsize = self.accelerator.unwrap_model(
            self.model).config.short_term_ctx_winsize

        if self.clean_context_ratio is None:
            context_mask = torch.zeros((batch_size, num_frames), device=latents.device).bool()
        else:
            if self.training_type == 'long_context':
                context_mask = torch.ones((batch_size, num_frames - short_term_ctx_winsize), device=latents.device).bool()
                noise_mask = torch.rand((batch_size, short_term_ctx_winsize), device=latents.device) < self.clean_context_ratio
                context_mask = torch.cat([context_mask, noise_mask], dim=-1)
                assert self.context_timestep_idx == -1
                timesteps[context_mask] = self.context_timestep_idx
                noisy_latents[context_mask] = latents[context_mask]  # clean context
            else:
                context_mask = torch.rand((batch_size, num_frames), device=latents.device) < self.clean_context_ratio
                assert self.context_timestep_idx == -1
                timesteps[context_mask] = self.context_timestep_idx
                noisy_latents[context_mask] = latents[context_mask]  # clean context

        model_pred = self.model(noisy_latents, timestep=timesteps, conditions=conditions).sample
        target = noise - latents

        model_pred, target, context_mask = \
            model_pred[:, -short_term_ctx_winsize:], target[:, -short_term_ctx_winsize:], context_mask[:, -short_term_ctx_winsize:]

        loss = torch.mean(((model_pred.float() - target.float())**2).reshape(target.shape[0], target.shape[1], -1), -1)

        loss_mask = ~context_mask
        loss = (loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-9)
        loss = loss.mean()

        return {'total_loss': loss}

    @torch.no_grad()
    def sample(self, val_dataloader, opt, wandb_logger=None, global_step=0):
        model = self.accelerator.unwrap_model(self.model)

        if self.ema is not None:
            self.ema.store(model)
            self.ema.copy_to(model)

        self.vae.eval()
        self.vae.enable_slicing()
        model.eval()

        val_pipeline = build_pipeline(opt['val']['val_pipeline'])(
            vae=self.vae,
            transformer=model,
            scheduler=copy.deepcopy(self.scheduler),
        )
        val_pipeline.execution_device = self.accelerator.device
        val_pipeline.set_progress_bar_config(disable=True)

        for batch_idx, batch in enumerate(tqdm(val_dataloader)):
            num_trajectory = opt['val']['sample_cfg']['sample_trajectory_per_video']
            gt_video = batch['video'].unsqueeze(1).repeat((1, num_trajectory, 1, 1, 1, 1))

            if 'select_last_frame' in opt['val']['sample_cfg']:
                gt_video = gt_video[:, :, -opt['val']['sample_cfg']['select_last_frame']:]
                batch['action'] = batch['action'][:, -opt['val']['sample_cfg']['select_last_frame']:]

            gt_video = rearrange(gt_video, 'b n t c h w -> (b n) t c h w')
            context_sequence = gt_video[:, :opt['val']['sample_cfg']['context_length']].clone()

            if 'label' in batch:
                conditions = {'label': batch['label']}
            elif 'action' in batch:
                conditions = {'action': batch['action']}
            else:
                conditions = None

            input_params = {
                'conditions': conditions,
                'context_sequence': context_sequence,
                'context_timestep_idx': self.context_timestep_idx,
                'unroll_length': opt['val']['sample_cfg']['unroll_length'],
                'num_inference_steps': opt['val']['sample_cfg']['num_inference_steps'],
                'guidance_scale': opt['val']['sample_cfg']['guidance_scale'],
                'sample_size': opt['val']['sample_cfg']['sample_size'],
                'use_kv_cache': opt['val']['sample_cfg'].get('use_kv_cache', True)
            }

            pred_video = val_pipeline.generate(**input_params)

            pred_video = rearrange(pred_video, '(b n) f c h w -> b n f c h w', n=num_trajectory)
            gt_video = rearrange(gt_video, '(b n) f c h w -> b n f c h w', n=num_trajectory)

            log_paired_video(
                sample=pred_video,
                gt=gt_video,
                context_frames=opt['val']['sample_cfg']['context_length'],
                save_suffix=batch['index'],
                save_dir=os.path.join(opt['path']['visualization'], f'iter_{global_step}'),
                wandb_logger=wandb_logger if batch_idx == 0 else None,  # only log first batch samples
                wandb_cfg={
                    'namespace': 'eval_vis',
                    'step': global_step,
                },
                annotate_context_frame=opt['val']['sample_cfg'].get('anno_context', True))

        if self.ema is not None:
            self.ema.restore(model)

        self.vae.disable_slicing()

    def read_video_folder(self, video_dir, num_trajectory):
        video_path_list = sorted(glob(os.path.join(video_dir, '*.mp4')))
        video_list = []
        for video_path in video_path_list:
            try:
                video = EncodedVideo.from_path(video_path, decode_audio=False)
                video = video.get_clip(start_sec=0.0, end_sec=video.duration)['video']
                video_list.append(video)
            except:
                print(f'error when opening {video_path}')

        videos = torch.stack(video_list)
        videos = rearrange(videos, 'b c (n f) h w -> b n f c h w', n=num_trajectory)

        videos = videos / 255.0
        videos_sample, videos_gt = torch.chunk(videos, 2, dim=-1)

        return videos_sample, videos_gt

    def eval_performance(self, opt, global_step=0):
        logger = get_logger('far', log_level='INFO')
        sample_dir = os.path.join(opt['path']['visualization'], f'iter_{global_step}')
        logger.info(f'begin evaluate {sample_dir}')

        video_metric = VideoMetric(metric=opt['val']['eval_cfg']['metrics'], device=self.accelerator.device)

        videos_sample, videos_gt = self.read_video_folder(sample_dir, num_trajectory=opt['val']['sample_cfg']['sample_trajectory_per_video'])

        logger.info(f'evaluating: sample of shape {videos_sample.shape}, gt of shape {videos_gt.shape}')
        result_dict = video_metric.compute(videos_sample.contiguous(), videos_gt.contiguous(), context_length=opt['val']['sample_cfg']['context_length'])
        return result_dict
