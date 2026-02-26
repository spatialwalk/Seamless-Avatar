import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    
def get_motion_dim(motion_type):
    if motion_type == 'gesture':
        nfeats = 13*6
    elif motion_type == 'expression':
        nfeats = 50+6
    elif motion_type == 'hands':
        nfeats = 15*2*6
    else:
        raise ValueError(f'invalid motion_type: {motion_type}')
    return nfeats


def print_model_trainable_params(model):
    # 新增代码：验证并打印可训练的参数

    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"\n可训练参数数量: {trainable_params/1e6:.2f}M || "
        f"总参数数量: {total_params/1e9:.2f}B || "
        f"可训练参数占比: {100 * trainable_params / total_params:.4f}%"
    )
    # 验证代码结束

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] device: {model.device}")
    print(f"[model] total params: {total_params / 1e9:.2f}B")

    return


def move_model_to_device_and_print_info(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print_model_trainable_params(model)
