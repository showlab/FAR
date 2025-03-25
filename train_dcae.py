import argparse
import gc
import os
import shutil

import torch
import torch.utils.checkpoint
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from far.data import build_dataset
from far.trainers import build_trainer
from far.utils.logger_util import MessageLogger, dict2str, reduce_loss_dict, set_path_logger, setup_wandb


def train(args):

    # load config
    opt = OmegaConf.to_container(OmegaConf.load(args.opt), resolve=True)

    # set accelerator
    accelerator = Accelerator(mixed_precision=opt['mixed_precision'], kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    # set experiment dir
    with accelerator.main_process_first():
        set_path_logger(accelerator, args.opt, opt, is_train=True)

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
    train_pipeline = build_trainer(opt['train']['train_pipeline'])(**opt['models'], accelerator=accelerator)

    # set optimizer
    train_opt = opt['train']
    optim_g_type, optim_d_type = train_opt['optim_g'].pop('type'), train_opt['optim_d'].pop('type')
    assert optim_g_type == optim_d_type == 'AdamW', 'only support AdamW now'
    G_params_to_optimize, D_params_to_optimize = train_pipeline.get_params_to_optimize(train_opt['param_names_to_optimize'])
    optimizer_g = torch.optim.AdamW(G_params_to_optimize, **train_opt['optim_g'])
    optimizer_d = torch.optim.AdamW(D_params_to_optimize, **train_opt['optim_d'])

    # Get the training dataset
    trainset_cfg = opt['datasets']['train']
    train_dataset = build_dataset(trainset_cfg)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=trainset_cfg['batch_size_per_gpu'], shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

    if opt['datasets'].get('sample'):
        sampleset_cfg = opt['datasets']['sample']
        sample_dataset = build_dataset(sampleset_cfg)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=sampleset_cfg['batch_size_per_gpu'], shuffle=False)
    else:
        sample_dataloader = None

    # Prepare learning rate scheduler in accelerate config
    total_batch_size = opt['datasets']['train']['batch_size_per_gpu'] * accelerator.num_processes

    num_training_steps = total_iter = opt['train']['total_iter']
    num_warmup_steps = opt['train']['warmup_iter']

    if opt['train']['lr_scheduler'] == 'constant_with_warmup':
        lr_scheduler_g = get_constant_schedule_with_warmup(
            optimizer=optimizer_g,
            num_warmup_steps=num_warmup_steps,
        )
        lr_scheduler_d = get_constant_schedule_with_warmup(
            optimizer=optimizer_d,
            num_warmup_steps=num_warmup_steps,
        )
    elif opt['train']['lr_scheduler'] == 'cosine_with_warmup':
        lr_scheduler_g = get_cosine_schedule_with_warmup(
            optimizer=optimizer_g,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        lr_scheduler_d = get_cosine_schedule_with_warmup(
            optimizer=optimizer_d,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        raise NotImplementedError

    # Prepare everything with our `accelerator`.
    train_pipeline.model, train_pipeline.discriminator, optimizer_g, optimizer_d, train_dataloader, sample_dataloader, lr_scheduler_g, lr_scheduler_d = accelerator.prepare(  # noqa: E501
        train_pipeline.model, train_pipeline.discriminator, optimizer_g, optimizer_d, train_dataloader, sample_dataloader, lr_scheduler_g, lr_scheduler_d)

    # set ema after prepare everything: sync the model init weight in ema
    train_pipeline.set_ema_model(ema_decay=opt['train'].get('ema_decay'))

    # Train!
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f"  Instantaneous batch size per device = {opt['datasets']['train']['batch_size_per_gpu']}")
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Total optimization steps = {total_iter}')

    if opt['path'].get('pretrain_network', None):
        load_path = opt['path'].get('pretrain_network')
    else:
        load_path = opt['path']['models']

    global_step = resume_checkpoint(args, accelerator, load_path, train_pipeline)

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)

    msg_logger = MessageLogger(opt, global_step)

    while global_step < total_iter:
        batch = next(train_data_yielder)
        """************************* start of an iteration*******************************"""
        # update generator loss
        loss_dict = train_pipeline.train_step(batch, iters=global_step)
        accelerator.backward(loss_dict['total_loss'])

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(train_pipeline.model.parameters(), opt['train']['max_grad_norm'])

        optimizer_g.step()
        lr_scheduler_g.step()
        optimizer_g.zero_grad()

        optimizer_d.step()
        lr_scheduler_d.step()
        optimizer_d.zero_grad()
        """************************* end of an iteration*******************************"""

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:

            if train_pipeline.ema is not None:
                train_pipeline.ema.step(accelerator.unwrap_model(train_pipeline.model))

            global_step += 1

            if global_step % opt['logger']['print_freq'] == 0:

                log_dict = reduce_loss_dict(accelerator, loss_dict)
                log_vars = {'iter': global_step}
                log_vars.update({'lrs': lr_scheduler_g.get_last_lr()})
                log_vars.update(log_dict)
                msg_logger(log_vars)

                if accelerator.is_main_process and wandb_logger:
                    wandb_log_dict = {
                        f'train/{k}': v
                        for k, v in log_vars.items()
                    }
                    wandb_log_dict['train/lrs_g'] = lr_scheduler_g.get_last_lr()[0]
                    wandb_log_dict['train/lrs_d'] = lr_scheduler_d.get_last_lr()[0]
                    wandb_logger.log(wandb_log_dict, step=global_step)

            if global_step % opt['val']['val_freq'] == 0 or global_step == total_iter:

                if sample_dataloader is not None:
                    train_pipeline.sample(sample_dataloader, opt, wandb_logger=wandb_logger, global_step=global_step)

                accelerator.wait_for_everyone()

                if accelerator.is_main_process and 'eval_cfg' in opt['val']:
                    result_dict = train_pipeline.eval_performance(opt, global_step=global_step)
                    logger.info(result_dict)

                    if wandb_logger:
                        wandb_log_dict = {
                            f'eval/{k}': v
                            for k, v in result_dict.items()
                        }
                        wandb_logger.log(wandb_log_dict, step=global_step)

                accelerator.wait_for_everyone()
                gc.collect()
                torch.cuda.empty_cache()

            if accelerator.is_main_process and (global_step % opt['logger']['save_checkpoint_freq'] == 0 or global_step == total_iter):
                save_checkpoint(args, logger, accelerator, train_pipeline, global_step, opt['path']['models'])


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
                ema_state = torch.load(os.path.join(output_dir, path, 'ema.pth'), weights_only=True)

                ema_state_dict = train_pipeline.ema.state_dict()
                model_state_dict = accelerator.unwrap_model(train_pipeline.model).state_dict()

                for key in set(model_state_dict.keys()) - set(ema_state_dict.keys()):
                    del ema_state[key]

                train_pipeline.ema.load_state_dict(ema_state)

    return global_step


def save_checkpoint(args, logger, accelerator, train_pipeline, global_step, output_dir):
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(f'{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints')
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = os.path.join(output_dir, f'checkpoint-{global_step}')
    accelerator.save_state(save_path)
    logger.info(f'Saved state to {save_path}')

    if train_pipeline.ema is not None:
        ema_state_dict = train_pipeline.ema.state_dict()
        model_state_dict = accelerator.unwrap_model(train_pipeline.model).state_dict()

        for key in set(model_state_dict.keys()) - set(ema_state_dict.keys()):
            ema_state_dict[key] = model_state_dict[key]

        torch.save(ema_state_dict, os.path.join(save_path, 'ema.pth'))
        logger.info(f'Saved ema model to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default='latest')
    parser.add_argument('--checkpoints_total_limit', type=int, default=None, help=('Max number of checkpoints to store.'))
    args = parser.parse_args()

    train(args)
