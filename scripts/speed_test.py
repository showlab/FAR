import json
import time

import decord
import numpy as np
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from far.models import build_model
from far.pipelines.pipeline_far import FARPipeline

decord.bridge.set_bridge('torch')


def build_inference_pipeline(model_cfg, pipeline=FARPipeline, device='cuda', weight_dtype=torch.bfloat16):
    # build model
    if model_cfg['transformer'].get('from_pretrained'):
        raise NotImplementedError
    else:
        init_cfg = model_cfg['transformer']['init_cfg']
        model = build_model(init_cfg['type'])(**init_cfg.get('config', {}))
        if init_cfg.get('pretrained_path'):
            state_dict = torch.load(init_cfg['pretrained_path'], map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)

    if model_cfg['vae'].get('from_pretrained'):
        raise NotImplementedError
    elif model_cfg['vae'].get('from_config'):
        with open(model_cfg['vae']['from_config'], 'r') as fr:
            config = json.load(fr)
        vae = build_model(model_cfg['vae']['type']).from_config(config)
        if model_cfg['vae'].get('from_config_pretrained'):
            state_dict_path = model_cfg['vae']['from_config_pretrained']
            state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
            vae.load_state_dict(state_dict)
    else:
        raise NotImplementedError

    if model_cfg['scheduler']['from_pretrained']:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_cfg['scheduler']['from_pretrained'], subfolder='scheduler')
    else:
        raise NotImplementedError

    model.requires_grad_(False).to(device, dtype=weight_dtype).eval()
    vae.requires_grad_(False).to(device, dtype=weight_dtype).eval()

    pipeline = pipeline(transformer=model, vae=vae, scheduler=scheduler)
    pipeline.execution_device = device

    return pipeline


def measure_time(func, *args, warmup=0, repeat=1, num_generate_frames=1, **kwargs):
    """
    Measure the execution time of a function (with CUDA sync),
    and print memory usage.
    """
    torch.cuda.reset_peak_memory_stats()

    # Warm-up runs
    for i in range(warmup):
        _ = func(*args, **kwargs)
        torch.cuda.synchronize()

    total_time = 0.0
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.time()
        _ = func(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time()
        total_time += (end - start)

    avg_time = total_time / repeat

    allocated_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    print(f'[TIMER] Generated {num_generate_frames} frames at {avg_time:.2f} s.')
    print(f'[MEMORY] Allocated: {allocated_mem:.2f} MB | Peak: {peak_mem:.2f} MB')


def read_video_dmlab(video_path):
    data = np.load(video_path)

    video = data['video']
    actions = data['actions']

    video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()
    actions = torch.from_numpy(actions)

    return video, actions


def benchmark_farb(batch_size, kv_cache=True):
    model_cfg = {
        'transformer': {
            'init_cfg': {
                'type': 'FAR_B_Long',
                'config': {
                    'short_term_ctx_winsize': 256,
                    'condition_cfg': {
                        'type': 'action',
                        'num_action_classes': 4
                    }
                },
            }
        },
        'vae': {
            'type': 'MyAutoencoderDC',
            'from_config': 'options/model_cfg/dcae/model_8x_c32_config.json',
        },
        'scheduler': {
            'from_pretrained': 'options/model_cfg/far/scheduler_config.json'
        }
    }

    # prepare pipeline
    device = torch.device('cuda')
    pipeline = build_inference_pipeline(model_cfg, pipeline=FARPipeline, device=device)
    pipeline.set_progress_bar_config(disable=True)

    # prepare_video
    ref_video, ref_actions = read_video_dmlab('assets/example/5.npz')
    ref_video = ref_video.to(device, dtype=torch.bfloat16)
    ref_actions = ref_actions.to(device)

    print(f'benchmark far-b with kv_cache={kv_cache}')

    for num_generate_frames in [16, 64, 128, 196, 256]:

        input_params = {
            'conditions': {'action': ref_actions[:num_generate_frames].unsqueeze(0).repeat((batch_size, 1))},
            'context_sequence': ref_video[:0, ...].unsqueeze(0).repeat((batch_size, 1, 1, 1, 1)),
            'unroll_length': num_generate_frames,
            'num_inference_steps': 25,
            'guidance_scale': 1,
            'generator': torch.Generator('cuda').manual_seed(0),
            'sample_size': 8,
            'use_kv_cache': kv_cache,
            'show_progress': False
        }

        with torch.no_grad():
            _ = measure_time(pipeline.generate, **input_params, num_generate_frames=num_generate_frames)


def benchmark_farb_long(batch_size, kv_cache=True):
    model_cfg = {
        'transformer': {
            'init_cfg': {
                'type': 'FAR_B_Long',
                'config': {
                    'short_term_ctx_winsize': 16,
                    'condition_cfg': {
                        'type': 'action',
                        'num_action_classes': 4
                    }
                },
            }
        },
        'vae': {
            'type': 'MyAutoencoderDC',
            'from_config': 'options/model_cfg/dcae/model_8x_c32_config.json',
        },
        'scheduler': {
            'from_pretrained': 'options/model_cfg/far/scheduler_config.json'
        }
    }

    # prepare pipeline
    device = torch.device('cuda')
    pipeline = build_inference_pipeline(model_cfg, pipeline=FARPipeline, device=device)
    pipeline.set_progress_bar_config(disable=True)

    # prepare_video
    ref_video, ref_actions = read_video_dmlab('assets/example/5.npz')
    ref_video = ref_video.to(device, dtype=torch.bfloat16)
    ref_actions = ref_actions.to(device)

    print(f'benchmark far-b-long with kv_cache={kv_cache}')

    for num_generate_frames in [16, 64, 128, 196, 256]:

        input_params = {
            'conditions': {'action': ref_actions[:num_generate_frames].unsqueeze(0).repeat((batch_size, 1))},
            'context_sequence': ref_video[:0, ...].unsqueeze(0).repeat((batch_size, 1, 1, 1, 1)),
            'unroll_length': num_generate_frames,
            'num_inference_steps': 25,
            'guidance_scale': 1,
            'generator': torch.Generator('cuda').manual_seed(0),
            'sample_size': 8,
            'use_kv_cache': kv_cache,
            'show_progress': False
        }

        with torch.no_grad():
            _ = measure_time(pipeline.generate, **input_params, num_generate_frames=num_generate_frames)


if __name__ == '__main__':
    batch_size = 4

    benchmark_farb(kv_cache=False, batch_size=batch_size)
    benchmark_farb(kv_cache=True, batch_size=batch_size)

    benchmark_farb_long(kv_cache=False, batch_size=batch_size)
    benchmark_farb_long(kv_cache=True, batch_size=batch_size)
