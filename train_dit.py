
import torch
import torch.nn as nn
from utils.util_func import seed_everything
from src.optim.scheduler import GradualWarmupScheduler
from src.engine.train_model import train_model, TrainConfig
from src.DiT.dyadic_model import DyadicTalkingHead, DyadicTalkingHeadConfig
import os
from src.engine.logger import get_logger
from icecream import ic, install
from configs import OUTPUT_ROOT_DIR, N_MOTIONS_FOR_DIT
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from src.data.motion_dataset import MotionDataset
from src.engine.util_ddp import init_distributed, cleanup
from utils.util_func import get_motion_dim


install()


def main():

    # -----------------------------
    seed_everything(42)
    exp_name_prefix = 'DiT_0107'
    motion_type = 'hands'  # expression, gesture, hands

    DEBUG = True

    # ----------------------------

    exp_name = f'{exp_name_prefix}_{motion_type}'
    ddp_args = init_distributed(use_ddp=False)

    # If use_swanlab=True, you will be required to log in SwanLab when running the code, and the logs will be uploaded to SwanLab.
    # swanlab is a simliar tool like wandb, and its website is https://swanlab.cn/
    logger = get_logger(use_swanlab=False, exp_name=exp_name)

    # -----------------------------
    batch_size = 2  # 32
    num_workers = 1  # 4
    train_dataset = MotionDataset(
        split='train',
        debug=DEBUG,
        motion_type=motion_type,
        chunksize=N_MOTIONS_FOR_DIT,
    )
    val_dataset = MotionDataset(
        split='dev',
        debug=DEBUG,
        motion_type=motion_type,
        chunksize=N_MOTIONS_FOR_DIT,
    )
    if ddp_args.distributed:
        # 使用 DistributedSampler 确保每个进程加载不同数据切片
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=ddp_args.world_size, rank=ddp_args.local_rank)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                  sampler=train_sampler, shuffle=False)

        val_sampler = DistributedSampler(
            val_dataset, num_replicas=ddp_args.world_size, rank=ddp_args.local_rank)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                                sampler=val_sampler, shuffle=False)
    else:
        # 单机情形直接常规 DataLoader
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

    ddp_args.train_sampler = train_sampler if ddp_args.distributed else None

    # -----------------------------
    talk_model_config = DyadicTalkingHeadConfig(
        motion_dim=get_motion_dim(motion_type)
    )
    model = DyadicTalkingHead(talk_model_config).to(ddp_args.local_rank)
    # batch = next(iter(train_loader))
    # for k,v in batch.items():
    #     print(v.shape,k)
    # output = model(batch)

    if ddp_args.distributed:
        # device_ids 和 output_device 必须是当前进程对应的GPU ID
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[ddp_args.local_rank],
            output_device=ddp_args.local_rank,
            find_unused_parameters=True,  # 临时手段
        )

    # # -----------------------------
    train_config = TrainConfig(
        num_epochs=500,
        ckpt_interval=10,
        eval_interval=5,
        warmup_epochs=20,
        train_log_iter_interval=100,
        view_loss_weight_dict_and_exit=False,
        exp_dir=os.path.join(OUTPUT_ROOT_DIR, exp_name),
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    warmup_steps = train_config.warmup_epochs * len(train_loader)
    scheduler = GradualWarmupScheduler(optimizer, total_epoch=warmup_steps)

    # -----------------------------
    start_epoch = 0
    train_model(train_config, model, train_loader,
                val_loader, start_epoch, optimizer, scheduler, logger, verbose=True, ddp_args=ddp_args)

    # 5. 清理
    if ddp_args.distributed:
        cleanup()

    logger.finish()

    return


if __name__ == "__main__":

    main()
