import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .util_audio_encoder import pad_audio
from .hubert import HubertModel
from configs import SAMPLE_RATE, FPS


def get_audio_encoder(audio_model_name, *args, **kwargs):
    if audio_model_name.lower() == 'hubert':
        return HubertEncoder(*args, **kwargs)
    else:
        raise ValueError(f'Unknown audio model name: {audio_model_name}')


class BaseAudioEncoder(nn.Module):
    """
    An abstract base class that defines a common interface for all audio encoders.
    It handles the common projection layer logic.
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, audio, frame_num):
        """Defines the forward pass logic, implemented by subclasses."""
        pass


class HubertEncoder(BaseAudioEncoder):
    """Concrete implementation for the Hubert model."""

    def __init__(self, feature_dim, fps=FPS, full_finetune=False):
        super().__init__(feature_dim)
        self.audio_encoder = HubertModel.from_pretrained(
            'facebook/hubert-base-ls960')
        if not full_finetune:
            self._freeze_parameters()  # freeze the encoder parameters
        self.projection = nn.Linear(
            self.audio_encoder.config.hidden_size, self.feature_dim)
        self.fps = fps

    def _freeze_parameters(self):
        self.audio_encoder.feature_extractor._freeze_parameters()

        frozen_layers = [0, 1]
        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_layers:
                    param.requires_grad = False
        return

    def forward(self, audio, frame_num, ratio=1):
        """Feature extraction logic for Hubert, including two strategies."""
        # # Strategy 1: resample during audio feature extraction
        # hidden_states = self.audio_encoder(pad_audio(audio), self.fps, frame_num=frame_num).last_hidden_state  # (N, L, 768)

       # Strategy 2: resample after audio feature extraction (BackResample)
        assert abs(audio.shape[1]/SAMPLE_RATE * FPS -
                   frame_num*ratio) <= 10, f'audio.shape[1]/SAMPLE_RATE * FPS: {audio.shape[1]/SAMPLE_RATE * FPS}, frame_num: {frame_num*ratio}'
        hidden_states = self.audio_encoder(pad_audio(audio), self.fps,
                                           frame_num=frame_num * 2).last_hidden_state  # (N, 2L, 768)
        hidden_states = hidden_states.transpose(1, 2)  # (N, 768, 2L)
        hidden_states = F.interpolate(
            hidden_states, size=frame_num, align_corners=False, mode='linear')  # (N, 768, L)
        hidden_states = hidden_states.transpose(1, 2)  # (N, L, 768)

        return self.projection(hidden_states)


# ===================================================================
# 5. 测试代码
# ===================================================================
if __name__ == "__main__":
    from icecream import ic
    from utils.util_func import seed_everything
    seed_everything(42)
    bz = 4
    n_samples = 17860
    frame_num = int(n_samples/SAMPLE_RATE*FPS)
    ic(frame_num)

    audio_encoder = get_audio_encoder('hubert', feature_dim=512)
    audio_encoder.eval()
    audio_feat = audio_encoder(torch.zeros(bz, n_samples), frame_num=frame_num)
    ic(audio_feat.shape)
    ic(audio_feat.sum())
