import torch
import torch.nn as nn
from tqdm import tqdm


class FlowMatching(nn.Module):
    def __init__(self, vector_field_net):
        """
        Flow Matching 采样器，封装训练和采样过程

        Args:
            vector_field_net: 预测向量场的网络 (类似于DDPM中的denoising_net)
        """
        super().__init__()
        self.vector_field_net = vector_field_net

    @property
    def device(self):
        return next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')

    def get_train_tuple(self, x1):
        """
        为训练准备数据元组，基于Conditional Flow Matching (CFM)

        Args:
            x1: 原始真实数据 (N, ...) e.g., (N, L, d_motion)

        Returns:
            t: 采样的时间步 (N,)
            xt: 在时间t的插值样本 (N, ...)
            ut: 目标向量场 (N, ...)
        """
        batch_size = x1.shape[0]

        # 1. 从标准正态分布中采样 x0
        x0 = torch.randn_like(x1)

        # 2. 从 [0, 1] 均匀采样时间 t
        # 添加一个小的 epsilon 防止 t=0 的情况，有时这有助于训练稳定性
        t = torch.rand(batch_size, device=x1.device) * (1 - 1e-4) + 1e-4

        # 调整 t 的形状以进行广播
        view_dims = (batch_size,) + (1,) * (x1.dim() - 1)
        t_broadcast = t.view(*view_dims)

        # 3. 计算在时间 t 的插值样本 xt
        # xt = t * x1 + (1 - t) * x0
        xt = t_broadcast * x1 + (1 - t_broadcast) * x0

        # 4. 目标向量场 ut (Conditional target)
        # ut = x1 - x0
        ut = x1 - x0

        return t, xt, ut

    @torch.no_grad()
    def sample(self, sample_shape, num_steps, guidance_scale, *args, **kwargs):
        """
        反向过程：通过求解ODE从噪声生成样本

        Args:
            sample_shape: 样本的形状 (batch_size, ...)
            num_steps: ODE求解器的步数
            guidance_scale: Classifier-Free Guidance (CFG) 的尺度。
                              如果 > 1.0，则启用CFG。
                              网络需要支持CFG（通常通过一个条件丢弃机制）。
            *args, **kwargs: 传递给 vector_field_net 的额外条件参数

        Returns:
            生成的样本 (N, ...)
        """

        # 1. 初始化随机噪声 x_0
        x0 = torch.randn(sample_shape, device=self.device)

        # 2. 创建求解ODE的时间步
        time_steps = torch.linspace(0, 1, num_steps + 1, device=self.device)

        # 3. 使用简单的欧拉法求解ODE
        # x_t+dt = x_t + v(x_t, t) * dt
        x_t = x0

        # 使用tqdm显示进度条
        for i in range(num_steps):  # , desc="Flow Matching Sampling"):
            t_current = time_steps[i]
            t_next = time_steps[i + 1]
            dt = t_next - t_current

            # 准备时间步输入
            step_in = t_current.expand(sample_shape[0])

            # 预测向量场 v(x_t, t)
            if guidance_scale > 1.0:
                # 预测向量场 v(x_t, t)

                # 1) 有条件预测
                v_cond = self.vector_field_net(
                    x_t, step_in, *args, **kwargs)

                # 2) 无条件预测 —— 通过 flag 告诉网络忽略条件
                kwargs_uncond = kwargs.copy()
                kwargs_uncond['force_unconditional'] = True
                v_uncond = self.vector_field_net(
                    x_t, step_in, *args, **kwargs_uncond)

                # 3) CFG 公式
                v = v_uncond + guidance_scale * (v_cond - v_uncond)

            else:
                # 无CFG，直接预测
                v = self.vector_field_net(x_t, step_in, *args, **kwargs)

            # 欧拉法更新步骤
            x_t = x_t + v * dt

        # 返回在 t=1 时的最终样本
        return x_t

    def forward_step(self, *args, **kwargs):
        """
        单步前向传播（用于训练），直接调用底层网络
        """
        return self.vector_field_net(*args, **kwargs)
