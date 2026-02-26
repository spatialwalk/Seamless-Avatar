import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List
from .dit_modules import DiTDecoderLayer
from .dit_modules import Qwen2_5_OmniAdaLayerNormZero_Final, Qwen2_5OmniDiTRotaryEmbedding, DiTTimestepEmbedding
from configs import FPS


@dataclass
class DiTDenoiseNetworkConfig:
    motion_dim: int = 56
    hidden_size: int = 512
    n_diff_steps: int = 500

    _attn_implementation: str = "sdpa"
    num_attention_heads: int = 8
    head_dim: int = 64
    ff_mult: int = 4
    dropout: float = 0.1

    num_blocks: int = 8
    block_size: int = FPS
    look_ahead_layers: List[int] = field(default_factory=lambda: [3])
    look_backward_layers: List[int] = field(default_factory=lambda: [0, 7])

    motion_emb_dim: int = 512

    p_use_learnable_audio: float = 0.5
    p_use_learnable_token: float = 0.5


class DiTDenoiseNetwork(nn.Module):

    def __init__(self, config=DiTDenoiseNetworkConfig()):
        super().__init__()

        self.rotary_embed = Qwen2_5OmniDiTRotaryEmbedding(config.head_dim)
        self.time_embed = DiTTimestepEmbedding(config.hidden_size)

        # set some configs
        self.block_size = config.block_size
        self.num_attention_heads = config.num_attention_heads

        input_dim = config.motion_dim + config.hidden_size
        self.proj_in = nn.Linear(input_dim, config.hidden_size)

        self.transformer_blocks = nn.ModuleList()
        for i in range(config.num_blocks):
            self.transformer_blocks.append(
                DiTDecoderLayer(
                    config,
                    look_ahead_block=1 if i in config.look_ahead_layers else 0,
                    look_backward_block=1 if i in config.look_backward_layers else 0,
                )
            )
        self.norm_out = Qwen2_5_OmniAdaLayerNormZero_Final(
            config.hidden_size)  # final modulation
        self.proj_out = nn.Linear(config.hidden_size, config.motion_dim)
        self.motion_dim = config.motion_dim

        self.learnable_audio_embed = nn.Parameter(
            torch.randn(1, 1, config.hidden_size))
        self.learnable_token_embed = nn.Parameter(
            torch.randn(1, 1, config.motion_emb_dim))
        self.p_use_learnable_audio = config.p_use_learnable_audio
        self.p_use_learnable_token = config.p_use_learnable_token

    @property
    def device(self):
        return next(self.parameters()).device

    def _create_block_diff(self, hidden_states):
        batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        block_indices = torch.arange(
            # [seq_length]
            seq_len, device=hidden_states.device) // self.block_size

        block_i = block_indices.unsqueeze(1)  # [seq_length, 1]
        block_j = block_indices.unsqueeze(0)  # [1, seq_length]
        block_diff = block_j - block_i  # (n, n)

        return block_diff.expand(batch, self.num_attention_heads, seq_len, seq_len)

    def forward(self,
                noised_motion_coef, time_step,
                audio_feat,
                force_unconditional=False,
                ):

        assert noised_motion_coef.shape[:2] == audio_feat.shape[:2]

        noAudio = False
        if self.training and torch.rand(1).item() < self.p_use_learnable_audio:
            noAudio = True

        if noAudio:
            audio_feat = self.learnable_audio_embed.expand(
                audio_feat.shape[0], audio_feat.shape[1], -1
            )

        if force_unconditional:
            audio_feat = self.learnable_audio_embed.expand(
                audio_feat.shape[0], audio_feat.shape[1], -1
            )

        input_feat = torch.cat(
            [noised_motion_coef, audio_feat], dim=-1)

        motion_feat = self.proj_in(input_feat)
        position_embeddings = self.rotary_embed(motion_feat)

        time_embedding = self.time_embed(time_step)
        cond_embedding = time_embedding  # + self.shape_proj(shape_coef)

        # Compute positional encodings
        blockwise_difference = self._create_block_diff(motion_feat)

        for transformer_block in self.transformer_blocks:
            motion_feat = transformer_block(
                motion_feat,
                cond_embedding,
                position_embeddings=position_embeddings,
                block_diff=blockwise_difference,
            )

        # 6. 将新的 cond_embedding 传递给最终的归一化层
        motion_feat = self.norm_out(motion_feat, cond_embedding)
        motion_pred = self.proj_out(motion_feat)
        return motion_pred
