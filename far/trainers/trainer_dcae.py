import json
import os
from glob import glob

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from einops import rearrange
from pytorchvideo.data.encoded_video import EncodedVideo
from tqdm import tqdm

from far.losses.lpips import LPIPS
from far.metrics.metric import VideoMetric
from far.models import build_model
from far.models.patch_discriminator import NLayerDiscriminator, calculate_adaptive_weight
from far.utils.ema_util import EMAModel
from far.utils.registry import TRAINER_REGISTRY
from far.utils.vis_util import log_paired_video


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


@TRAINER_REGISTRY.register()
class DCAETrainer:

    def __init__(
        self,
        accelerator,
        model_cfg,
        perceptual_weight=1.0,
        disc_weight=0,
        disc_start_iter=50001
    ):
        super(DCAETrainer, self).__init__()

        self.accelerator = accelerator
        weight_dtype = torch.float32
        if accelerator.mixed_precision == 'fp16':
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == 'bf16':
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        if model_cfg['vae'].get('from_config'):
            with open(model_cfg['vae']['from_config'], 'r') as fr:
                config = json.load(fr)
            self.model = build_model(
                model_cfg['vae']['type']).from_config(config)
        elif model_cfg['vae'].get('from_pretrained'):
            self.model = build_model(model_cfg['vae']['type']).from_pretrained(
                model_cfg['vae']['from_pretrained'])
        else:
            raise NotImplementedError
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.ema = None

        self.perceptual_weight = perceptual_weight
        self.disc_weight = disc_weight
        self.disc_start_iter = disc_start_iter

        if self.perceptual_weight > 0:
            self.perceptual_loss = LPIPS().to(accelerator.device).eval()

        if self.disc_weight > 0:
            self.discriminator = NLayerDiscriminator()
            self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)

    def set_ema_model(self, ema_decay):
        logger = get_logger('far', log_level='INFO')

        if ema_decay is not None:
            self.ema = EMAModel(self.accelerator.unwrap_model(self.model), decay=ema_decay)
            logger.info(f'enable EMA training with decay {ema_decay}')

    def get_params_to_optimize(self, param_names_to_optimize):
        logger = get_logger('far', log_level='INFO')

        G_params_to_optimize = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                G_params_to_optimize.append(param)
                logger.info(f'optimize params: {name}')

        logger.info(
            f'#Trained Generator Parameters: {sum([p.numel() for p in G_params_to_optimize]) / 1e6} M'
        )

        D_params_to_optimize = []
        for name, param in self.discriminator.named_parameters():
            if param.requires_grad:
                D_params_to_optimize.append(param)
                logger.info(f'optimize params: {name}')

        logger.info(
            f'#Trained Discriminator Parameters: {sum([p.numel() for p in D_params_to_optimize]) / 1e6} M'
        )

        return G_params_to_optimize, D_params_to_optimize

    def train_step_pixel_loss(self, batch, iters=-1):
        self.model.train()
        loss_dict = {}

        inputs = batch['video'].to(dtype=self.weight_dtype)
        inputs = inputs * 2 - 1

        if inputs.dim() == 5:  # video
            inputs = rearrange(inputs, 'b t c h w -> (b t) c h w')

        reconstructions = self.model(inputs, return_dict=False)[0]

        # reconstruction loss
        rec_loss = F.l1_loss(inputs, reconstructions)
        loss_dict['rec_loss'] = rec_loss

        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_weight * self.perceptual_loss(inputs, reconstructions)
            loss_dict['perceptual_loss'] = perceptual_loss

        total_loss = rec_loss + perceptual_loss
        loss_dict['total_loss'] = total_loss

        return loss_dict

    def train_step_gan_loss(self, batch, iters=-1):
        self.model.train()
        loss_dict = {}

        # train generator
        for p in self.discriminator.parameters():
            p.requires_grad = False

        for p in self.accelerator.unwrap_model(self.model).encoder.parameters():
            p.requires_grad = False

        inputs = batch['video'].to(dtype=self.weight_dtype)
        inputs = inputs * 2 - 1

        if inputs.dim() == 5:  # video
            inputs = rearrange(inputs, 'b t c h w -> (b t) c h w')

        reconstructions = self.model(inputs, return_dict=False)[0]

        # reconstruction loss
        rec_loss = F.l1_loss(inputs, reconstructions)
        loss_dict['rec_loss'] = rec_loss

        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_weight * self.perceptual_loss(inputs, reconstructions)
            loss_dict['perceptual_loss'] = perceptual_loss

        # generator loss
        logits_fake = self.discriminator(reconstructions)
        g_loss = -torch.mean(logits_fake)
        d_weight = self.disc_weight * calculate_adaptive_weight(
            rec_loss + perceptual_loss, g_loss, last_layer=self.accelerator.unwrap_model(self.model).get_last_layer())
        g_loss = d_weight * g_loss
        loss_dict['g_loss'] = g_loss

        total_loss_g = rec_loss + perceptual_loss + g_loss
        loss_dict['total_loss_g'] = total_loss_g

        # train discriminator

        for p in self.discriminator.parameters():
            p.requires_grad = True

        logits_fake = self.discriminator(reconstructions.detach())
        logits_real = self.discriminator(inputs.detach())

        total_loss_d = hinge_d_loss(logits_real, logits_fake)
        loss_dict['total_loss_d'] = total_loss_d

        loss_dict['total_loss'] = total_loss_g + total_loss_d
        return loss_dict

    def train_step(self, batch, iters=-1):
        if iters < self.disc_start_iter:
            return self.train_step_pixel_loss(batch, iters)
        else:
            return self.train_step_gan_loss(batch, iters)

    @torch.no_grad()
    def sample(self, val_dataloader, opt, wandb_logger=None, global_step=0):
        model = self.accelerator.unwrap_model(self.model)

        if self.ema is not None:
            self.ema.store(model)
            self.ema.copy_to(model)

        model.eval()

        for batch_idx, item in enumerate(tqdm(val_dataloader)):

            gt_video = 2 * item['video'] - 1
            gt_video = rearrange(gt_video, 'b t c h w -> (b t) c h w')

            recon_video = model(gt_video, return_dict=False)[0]
            recon_video = rearrange(recon_video, '(b t) c h w -> b 1 t c h w', b=item['video'].shape[0])
            recon_video = (recon_video + 1) / 2

            gt_video = rearrange(gt_video, '(b t) c h w -> b 1 t c h w', b=item['video'].shape[0])
            gt_video = (gt_video + 1) / 2

            # step 1: log generation video
            log_paired_video(
                sample=recon_video,
                gt=gt_video,
                context_frames=opt['val']['sample_cfg']['context_length'],
                save_suffix=item['index'],
                save_dir=os.path.join(opt['path']['visualization'], f'iter_{global_step}'),
                wandb_logger=wandb_logger,
                wandb_cfg={
                    'namespace': 'eval_vis',
                    'step': global_step,
                })

        if self.ema is not None:
            self.ema.restore(model)

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

        # filter out context frame
        videos_sample = videos_sample
        videos_gt = videos_gt
        return videos_sample, videos_gt

    def eval_performance(self, opt, global_step=0):
        logger = get_logger('far', log_level='INFO')
        sample_dir = os.path.join(opt['path']['visualization'], f'iter_{global_step}')
        logger.info(f'begin evaluate {sample_dir}')

        video_metric = VideoMetric(metric=opt['val']['eval_cfg']['metrics'], device=self.accelerator.device)

        videos_sample, videos_gt = self.read_video_folder(sample_dir, num_trajectory=1)
        logger.info(f'evaluating: sample of shape {videos_sample.shape}, gt of shape {videos_gt.shape}')
        result_dict = video_metric.compute(videos_sample.contiguous(), videos_gt.contiguous(), context_length=opt['val']['sample_cfg']['context_length'])
        return result_dict
