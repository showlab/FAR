import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers.models import AutoencoderKL, AutoencoderKLCogVideoX, DiTTransformer2DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm

from far.utils.registry import PIPELINE_REGISTRY


@PIPELINE_REGISTRY.register()
class FARPipeline(DiffusionPipeline):

    model_cpu_offload_seq = 'transformer->vae'

    def __init__(
        self,
        transformer: DiTTransformer2DModel,
        vae: AutoencoderKL,
        scheduler: FlowMatchEulerDiscreteScheduler
    ):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler)

    def vae_encode(self, context_sequence):
        # normalize: [0, 1] -> [-1, 1]
        context_sequence = context_sequence * 2 - 1

        batch_size = context_sequence.shape[0]
        context_sequence = rearrange(context_sequence, 'b t c h w -> (b t) c h w')
        if isinstance(self.vae, AutoencoderKL):
            context_sequence = self.vae.encode(context_sequence.to(dtype=self.vae.dtype)).latent_dist.sample()
        else:
            context_sequence = self.vae.encode(context_sequence.to(dtype=self.vae.dtype)).latent

        context_sequence = context_sequence * self.vae.config.scaling_factor
        context_sequence = rearrange(context_sequence, '(b t) c h w -> b t c h w', b=batch_size)
        return context_sequence

    def vae_decode(self, latents):
        batch_size = latents.shape[0]
        latents = 1 / self.vae.config.scaling_factor * latents

        if isinstance(self.vae, AutoencoderKLCogVideoX):
            latents = rearrange(latents, 'b t c h w -> b c t h w')
        else:
            latents = rearrange(latents, 'b t c h w -> (b t) c h w')

        samples = self.vae.decode(latents.to(dtype=self.vae.dtype)).sample

        if isinstance(self.vae, AutoencoderKLCogVideoX):
            samples = rearrange(samples, 'b c t h w -> b t c h w', b=batch_size)
        else:
            samples = rearrange(samples, '(b t) c h w -> b t c h w', b=batch_size)

        samples = (samples / 2 + 0.5).clamp(0, 1)
        return samples
    
    @torch.no_grad()
    def generate(
        self,
        unroll_length,
        guidance_scale,
        context_timestep_idx=-1,
        context_sequence=None,
        conditions=None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        sample_size=32,
        batch_size=1,
        use_kv_cache=True,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
    ):
        if context_sequence is None:
            current_context_length = 0
        else:
            batch_size, current_context_length = context_sequence.shape[0], context_sequence.shape[1]

        if current_context_length == 0:
            latents = None
        else:
            # step 1: encode vision context to embedding
            latents = self.vae_encode(context_sequence)

        latent_size = sample_size
        latent_channels = self.transformer.config.in_channels
        init_latents = randn_tensor(
            shape=(batch_size, unroll_length, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self.execution_device,
            dtype=self.vae.dtype,
        )

        if use_kv_cache:
            context_cache = {'is_cache_step': True, 'kv_cache': {}, 'cached_seqlen': 0, 'multi_level_cache_init': False}
        else:
            context_cache = {'is_cache_step': True, 'kv_cache': None, 'cached_seqlen': 0, 'multi_level_cache_init': False}

        for step in tqdm(range(current_context_length, current_context_length + unroll_length)):

            if conditions is not None and 'action' in conditions:
                step_condition = {'action': conditions['action'][:, :step + 1]}
            else:
                step_condition = copy.deepcopy(conditions)

            pred_latents, context_cache = self(
                conditions=step_condition,
                vision_context=latents,
                context_cache=context_cache,
                latents=init_latents[:, step - current_context_length: step - current_context_length + 1],
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                context_timestep_idx=context_timestep_idx)
            
            if step == 0:
                latents = pred_latents
            else:
                latents = torch.cat([latents, pred_latents], dim=1)

        samples = self.vae_decode(latents)
        return samples

    @torch.no_grad()
    def __call__(
        self,
        vision_context,
        latents,
        conditions=None,
        context_cache=None,
        context_timestep_idx=-1,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 50,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:

        batch_size = latents.shape[0]

        if conditions is not None:
            if 'label' in conditions:
                class_labels = conditions['label'].to(self.execution_device).reshape(-1)
                if guidance_scale > 1:
                    null_class_idx = self.transformer.config.condition_cfg['num_classes']
                    class_null = torch.tensor([null_class_idx] * batch_size, device=self.execution_device)
                    class_labels_input = torch.cat([class_null, class_labels], 0)
                else:
                    class_labels_input = class_labels
                conditions['label'] = class_labels_input
            elif 'action' in conditions:
                actions = conditions['action'].to(self.execution_device)
                if guidance_scale > 1:
                    null_action_idx = self.transformer.config.condition_cfg['num_action_classes']
                    action_null = torch.tensor([null_action_idx] * batch_size, device=self.execution_device)
                    action_null = action_null.unsqueeze(1).repeat((1, conditions['action'].shape[1]))
                    actions_input = torch.cat([action_null, actions], 0)
                elif guidance_scale == -1:  # unconditional
                    actions_input = torch.ones_like(actions) * self.transformer.condition_cfg['num_action_classes']
                else:
                    actions_input = actions
                conditions['action'] = actions_input
            else:
                raise NotImplementedError

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        context_cache['is_cache_step'] = True if vision_context is not None else False

        for t in self.progress_bar(self.scheduler.timesteps):
            timesteps = t

            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
            if guidance_scale > 1 and vision_context is not None:
                vision_context_input = torch.cat([vision_context] * 2)
            else:
                vision_context_input = vision_context

            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == 'mps'
                if isinstance(timesteps, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(latent_model_input.shape[0])
            timesteps = timesteps.unsqueeze(-1)

            # predict noise model_output
            if vision_context_input is not None:
                assert context_timestep_idx == -1, 'we use timestep=-1 to represent clean context'
                context_timesteps = torch.tensor([context_timestep_idx], dtype=timesteps.dtype, device=timesteps.device)

                context_timesteps = context_timesteps.expand(latent_model_input.shape[0])
                context_timesteps = context_timesteps.unsqueeze(-1).repeat((1, vision_context_input.shape[1]))
                timesteps = torch.cat([context_timesteps, timesteps], dim=-1)
                latent_model_input = torch.cat([vision_context_input, latent_model_input], dim=1)

            noise_pred, context_cache = self.transformer(
                latent_model_input,
                context_cache=context_cache,
                timestep=timesteps,
                conditions=conditions,
                return_dict=False)
            noise_pred = noise_pred.to(latent_model_input.dtype)

            context_cache['is_cache_step'] = False

            # perform guidance
            if guidance_scale > 1:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute previous image: x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents, context_cache
        
