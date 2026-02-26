from dataclasses import dataclass
import torch
import tqdm
import os
from icecream import ic
from .eval_model import eval_model
from .util_engine import save_checkpoint
from configs import OUTPUT_ROOT_DIR
from utils.util_func import move_model_to_device_and_print_info


@dataclass
class TrainConfig:
    eval_interval: int = 1
    ckpt_interval: int = 5
    num_epochs: int = 100
    warmup_epochs: int = 5
    train_log_iter_interval: int = 100
    view_loss_weight_dict_and_exit: bool = False
    exp_dir: str = f"{OUTPUT_ROOT_DIR}/exp_0"
    grad_accum_steps: int = 1


def train_model(train_config, model, train_loader,
                val_loader, start_epoch, optimizer, scheduler, logger, verbose=False, ddp_args=None):
    """Main training loop."""

    if ddp_args is None:
        move_model_to_device_and_print_info(model)

    ckpt_save_dir = os.path.join(train_config.exp_dir, 'checkpoints')
    os.makedirs(ckpt_save_dir, exist_ok=True)
    iter_num = 0
    assert train_config.ckpt_interval % train_config.eval_interval == 0

    for epoch in tqdm.tqdm(range(start_epoch, train_config.num_epochs + 1), desc="Total train progress"):
        # Evaluation and checkpointing are performed at the beginning of an epoch loop.
        # This means for epoch N, we're evaluating the model state from the end of epoch N-1.

        is_eval_epoch = epoch % train_config.eval_interval == 0
        is_ckpt_epoch = epoch % train_config.ckpt_interval == 0

        if is_eval_epoch:
            eval_loss_dict = eval_model(
                model, val_loader, epoch, logger, verbose)

            if train_config.view_loss_weight_dict_and_exit:
                ic(eval_loss_dict)
                return

        if is_ckpt_epoch:
            save_checkpoint(epoch, iter_num, model, scheduler,
                            eval_loss_dict, ckpt_save_dir)

        if epoch < train_config.num_epochs:
            if ddp_args is not None:
                if ddp_args.distributed:
                    ddp_args.train_sampler.set_epoch(epoch)
            iter_num = _train_one_epoch(
                epoch, model, train_loader, optimizer, scheduler, iter_num, train_config, logger, verbose)
    return


def _train_one_epoch(epoch, model, train_loader, optimizer, scheduler, iter_num, train_config, logger, verbose=False):
    """Trains the model for one epoch."""
    model.train()
    if verbose:
        progress_bar = tqdm.tqdm(train_loader, desc=f"Train epoch {epoch:03d}")
    else:
        progress_bar = train_loader

    for batch in progress_bar:
        if iter_num % train_config.grad_accum_steps == 0:
            optimizer.zero_grad()

        output = model(batch)
        loss_dict = output['loss_dict']
        total_loss = loss_dict['total_loss']
        total_loss = total_loss / train_config.grad_accum_steps

        total_loss.backward()

        if (iter_num + 1) % train_config.grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()

        if iter_num % train_config.train_log_iter_interval == 0:
            for key, value in loss_dict.items():
                if torch.isnan(value) or torch.isinf(value):
                    ic(key, value)
                    raise ValueError(f'{key} is nan or inf')

                logger.log({f'train/{key}': float(value)}, step=iter_num)
            logger.log(
                {'lr': optimizer.param_groups[0]['lr']}, step=iter_num)
        iter_num += 1

    return iter_num
