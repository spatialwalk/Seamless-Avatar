import numpy as np
import librosa
from configs import SAMPLE_RATE

def get_audio_processor(audio_model_name):
    """
    Factory function to get the appropriate audio processor.
    """
    if audio_model_name == 'hubert':
        return HubertAudioProcessor()
    else:
        raise ValueError(f"audio_model_name: {audio_model_name} not supported")


class HubertAudioProcessor():
    """
    Processes audio for Hubert (normalization).
    """

    def process(self, wav_path, sampling_rate=SAMPLE_RATE):
        audio_data, _ = librosa.load(wav_path, sr=sampling_rate)
        assert audio_data.ndim == 1, f"audio_data.ndim: {audio_data.ndim} != 1"
        # audio_mean = audio_data.mean()
        # audio_std = audio_data.std()
        # return (audio_data - audio_mean) / (audio_std + 1e-5)
        return audio_data
