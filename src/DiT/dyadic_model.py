import torch
import torch.nn as nn
from dataclasses import dataclass

from .modules.dit_denoise_network import DiTDenoiseNetwork
from ..audio_encoder.audio_encoder import get_audio_encoder
from ..losses.motion_loss import MotionLoss
from configs import N_MOTIONS_FOR_DIT, SAMPLE_RATE, FPS
import torch.nn.functional as F
from icecream import ic
from .utils_model.flow_matching import FlowMatching
from .modules.dit_denoise_network import DiTDenoiseNetworkConfig


@dataclass
class DyadicTalkingHeadConfig:
    motion_dim: int
    audio_model_name: str = "hubert"
    n_motions: int = N_MOTIONS_FOR_DIT
    feature_dim: int = 512


class DyadicTalkingHead(nn.Module):
    def __init__(self, config):

        super().__init__()
        self.audio_encoder = get_audio_encoder(
            config.audio_model_name, feature_dim=config.feature_dim)

        denoise_network_config = DiTDenoiseNetworkConfig(
            motion_dim=config.motion_dim)
        self.denoise_net = DiTDenoiseNetwork(denoise_network_config)

        self.flow_matching = FlowMatching(self.denoise_net)
        self.n_motions = config.n_motions

        self.loss_fn = MotionLoss()

        self.config = config

    @property
    def device(self):
        return next(self.parameters()).device

    def _preprocess_batch(self, batch):

        device = self.device

        motion_coef_gt = batch['motion'].to(device)
        audio = batch['audio'].to(device)

        assert motion_coef_gt.shape[
            1] % self.n_motions == 0, f"Motion coef shape: {motion_coef_gt.shape}, expected multiple of: {self.n_motions}"

        return motion_coef_gt, audio

    def forward(self, batch):
        motion_coef_gt, audio = self._preprocess_batch(
            batch)
        assert motion_coef_gt.shape[
            1] == self.n_motions, f"motion_coef_gt.shape[1]: {motion_coef_gt.shape[1]}, expected: {self.n_motions}"

        # Extract audio feature
        audio_feat_a = self.audio_encoder(
            audio, frame_num=self.n_motions
        )  # (N, L, feature_dim)

        # Apply diffusion noise and get prediction
        t, xt, ut = self.flow_matching.get_train_tuple(motion_coef_gt)
        ut_pred = self.denoise_net(
            xt, t, audio_feat_a)

        output = {
            'pred': ut_pred,
            'gt': ut
        }
        loss_x, loss_v = self.loss_fn(ut_pred, ut)
        output['loss_dict'] = {
            'loss_x': loss_x,
            'loss_v': loss_v,
            'total_loss': loss_x + loss_v
        }

        return output

    @torch.no_grad()
    def sample(self, audio, num_steps=10):
        assert audio.ndim == 1, f"audio.ndim: {audio.ndim} != 1"

        n_frames = int(audio.shape[0]/SAMPLE_RATE*FPS)
        audio = audio[None, ...].to(self.device)
        # Extract audio feature
        audio_feat = self.audio_encoder(
            audio, frame_num=n_frames
        )

        cfg_scale = 1.0

        sample_shape = (1, n_frames, self.config.motion_dim)

        motion_pred = self.flow_matching.sample(
            sample_shape,
            num_steps,
            cfg_scale,
            audio_feat
        )
        return motion_pred
