import argparse
import os

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from far.data import build_dataset
from far.trainers import build_trainer
from far.utils.logger_util import dict2str, set_path_logger, setup_wandb


def test(args):
    # load config
    opt = OmegaConf.to_container(OmegaConf.load(args.opt), resolve=True)

    # set accelerator
    accelerator = Accelerator(mixed_precision=opt['mixed_precision'])

    # set experiment dir
    with accelerator.main_process_first():
        set_path_logger(accelerator, args.opt, opt, is_train=False)

    # get logger
    logger = get_logger('far', log_level='INFO')
    logger.info(accelerator.state)
    logger.info(dict2str(opt))

    # get wandb
    if accelerator.is_main_process and opt['logger'].get('use_wandb', False):
        wandb_logger = setup_wandb(name=opt['name'], save_dir=opt['path']['log'])
    else:
        wandb_logger = None

    # If passed along, set the training seed now.
    if opt.get('manual_seed') is not None:
        set_seed(opt['manual_seed'] + accelerator.process_index)

    # load trainer pipeline
    train_pipeline = build_trainer(
        opt['train']['train_pipeline'])(**opt['models'], accelerator=accelerator)

    if opt['datasets'].get('sample'):
        sampleset_cfg = opt['datasets']['sample']
        sample_dataset = build_dataset(sampleset_cfg)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=sampleset_cfg['batch_size_per_gpu'], shuffle=False)
    else:
        sample_dataloader = None

    # Prepare everything with our `accelerator`.
    train_pipeline.model, sample_dataloader = accelerator.prepare(train_pipeline.model, sample_dataloader)

    # set ema after prepare everything: sync the model init weight in ema
    train_pipeline.set_ema_model(ema_decay=opt['train'].get('ema_decay'))

    # Test!
    logger.info('***** Running testing *****')

    if opt['path']['pretrain_network']:
        global_step = resume_checkpoint(args, accelerator, os.path.join(opt['path']['pretrain_network'], 'models'), train_pipeline)
    else:
        logger.info('infer only pretrained model without load checkpoint!')
        global_step = 0

    logger.info(f'begin evaluation step-{global_step}:')

    if sample_dataloader is not None:
        train_pipeline.sample(sample_dataloader, opt, wandb_logger=wandb_logger, global_step=global_step)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and 'eval_cfg' in opt['val']:
        result_dict = train_pipeline.eval_performance(opt, global_step=global_step)
        logger.info(result_dict)

        if wandb_logger:
            wandb_log_dict = {f'eval/{k}': v for k, v in result_dict.items()}
            wandb_logger.log(wandb_log_dict, step=global_step)


def resume_checkpoint(args, accelerator, output_dir, train_pipeline):
    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != 'latest':
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith('checkpoint')]
            dirs = sorted(dirs, key=lambda x: int(x.split('-')[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f'Resuming from checkpoint {path}')
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split('-')[1])

            if train_pipeline.ema is not None:
                accelerator.print(f'Resuming ema from checkpoint {path}')
                ema_state = torch.load(os.path.join(output_dir, path, 'ema.pth'), map_location='cpu', weights_only=True)
                train_pipeline.ema.load_state_dict(ema_state)
    return global_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default='latest')
    args = parser.parse_args()

    test(args)
