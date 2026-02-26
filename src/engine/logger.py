import swanlab
import os
from configs import OUTPUT_ROOT_DIR
import torch

def get_logger(use_swanlab=True, project_name='dyadic-interact',exp_name='baseline'):

    # 若当前进程不是主进程（rank 0），直接返回空记录器，
    # 避免多 GPU 场景下重复写日志 / 竞态
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return NoOpLogger()

    # True, False
    if use_swanlab:
        os.environ['SWANLAB_LOG_DIR'] = f'{OUTPUT_ROOT_DIR}/swanlog'
        swanlab_config = {
            "project": project_name,
            "experiment_name": exp_name,
            "description": "A baseline training run.",
            "mode": "cloud"  # cloud, disabled
        }
        logger = SwanLabLogger(**swanlab_config)
    else:
        logger = NoOpLogger()
    return logger


class BaseLogger:
    """日志记录器的基类。"""

    def log(self, data: dict, step: int):
        """
        记录指标。

        Args:
            data (dict): 要记录的指标字典。
            step (int): 当前的步数或轮次。
        """
        raise NotImplementedError

    def finish(self):
        """完成日志记录。"""
        pass


class SwanLabLogger(BaseLogger):
    """SwanLab 的日志记录器。"""

    def __init__(self, **kwargs):
        swanlab.init(**kwargs)

    def log(self, data: dict, step: int):
        swanlab.log(data, step=step)

    def finish(self):
        swanlab.finish()


class NoOpLogger(BaseLogger):
    """一个什么都不做的日志记录器，用于禁用日志。"""

    def log(self, data: dict, step: int):
        pass
