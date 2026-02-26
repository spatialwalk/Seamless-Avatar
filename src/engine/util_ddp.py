import os
import torch.distributed as dist
from icecream import ic
from argparse import Namespace


def setup(rank: int, world_size: int):
    """初始化进程组

    首选让 torchrun 注入的 MASTER_ADDR/MASTER_PORT 生效；
    若用户直接 python tmp.py 单机运行，则为其设置默认值并使用 env:// 初始化。
    """
    # 若用户未显式设置，提供安全默认值（IPv4 + 常用端口）
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    # 使用 env:// 让 PyTorch 从环境变量解析地址
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    """销毁进程组"""
    dist.destroy_process_group()


def init_distributed(use_ddp=True):
    # 1. 初始化/解析多进程信息
    # 在 torchrun 多卡场景下，这两个环境变量会被自动注入
    ddp_args = Namespace()

    if use_ddp:

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        ic(local_rank, world_size)

        distributed = world_size > 1
        if distributed:
            if dist.is_initialized():  # 已经初始化过就直接返回
                return
            setup(local_rank, world_size)
        ddp_args.distributed = distributed
        ddp_args.local_rank = local_rank
        ddp_args.world_size = world_size
    else:
        ddp_args.distributed = False
        ddp_args.local_rank = 0
        

    return ddp_args
