import torch
import os


def save_checkpoint(epoch, iter_num, model, scheduler, loss_dict, save_ckpt_folder):
    """
    在单机多卡 DDP 下只由 rank 0 保存模型。
    """
    # 若没初始化分布式，默认就认为当前进程是 rank 0
    is_main_process = (not torch.distributed.is_initialized()
                       ) or torch.distributed.get_rank() == 0
    if not is_main_process:
        return

    # 兼容普通模型 / DataParallel / DistributedDataParallel
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        {
            'model': model_state,
            'loss_dict': loss_dict,
            # 'scheduler': scheduler.state_dict(),
            # 'iter': iter_num,
            'epoch': epoch,
        },
        os.path.join(save_ckpt_folder, f'{epoch:03d}.pt')
    )
