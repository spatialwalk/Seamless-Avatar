import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class PersonTokenData:
    '''
    要确保每一个成员都是一个 torch.tensor
    '''
    expression_token_ids: torch.Tensor | None
    gesture_token_ids: torch.Tensor | None
    hands_token_ids: torch.Tensor | None
    audio: torch.Tensor | None
    text_input_ids: torch.Tensor | None
    text_attention_mask: torch.Tensor | None

    def __post_init__(self):

        assert self.text_input_ids.shape == self.text_attention_mask.shape
        assert self.text_input_ids.ndim == 2
        assert self.audio.ndim == 1
        assert self.expression_token_ids.ndim == 1
        assert self.gesture_token_ids.ndim == 1
        assert self.hands_token_ids.ndim == 1

        n_token_list = [
            self.expression_token_ids.shape[0],
            self.gesture_token_ids.shape[0],
            self.hands_token_ids.shape[0],
            self.text_input_ids.shape[0]
        ]
        assert max(n_token_list) - \
            min(n_token_list) == 0, f"n_token_list: {n_token_list}"
        self.n_tokens = min(n_token_list)

    def __repr__(self) -> str:
        lines = ["PersonData:"]
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            # , mean: {v.mean():.2f}, std: {v.std():.2f}")
            lines.append(f"{v.shape}\t{v.dtype}\t[{k}]")
        return "\n".join(lines)

    def to_dict(self):
        data_dict = {
            'expression_token_ids': self.expression_token_ids,
            'gesture_token_ids': self.gesture_token_ids,
            'hands_token_ids': self.hands_token_ids,
            'audio': self.audio,
            'text_input_ids': self.text_input_ids,
            'text_attention_mask': self.text_attention_mask,
        }

        return data_dict

    def split_to_chunks(self, chunksize_token, chunksize_audio):

        person_data_list = []

        chunks = math.ceil(self.n_tokens/chunksize_token)
        for i in range(chunks):

            token_start_id = i*chunksize_token
            token_end_id = (i+1)*chunksize_token

            audio_start_sample = i * chunksize_audio
            audio_end_sample = (i+1) * chunksize_audio

            tmp_audio = self.audio[audio_start_sample:audio_end_sample]
            tmp_text_input_ids = self.text_input_ids[token_start_id:token_end_id]
            tmp_text_attention_mask = self.text_attention_mask[token_start_id:token_end_id]
            tmp_expression_token_ids = self.expression_token_ids[token_start_id:token_end_id]
            tmp_gesture_token_ids = self.gesture_token_ids[token_start_id:token_end_id]
            tmp_hands_token_ids = self.hands_token_ids[token_start_id:token_end_id]

            # pad samples
            pad_samples = chunksize_audio - tmp_audio.shape[0]
            if pad_samples > 0:
                tmp_audio = F.pad(
                    tmp_audio, (0, pad_samples), mode='constant', value=0)

            # pad tokens
            pad_tokens = chunksize_token - tmp_text_input_ids.shape[0]

            if pad_tokens > 0:
                tmp_text_input_ids = F.pad(
                    tmp_text_input_ids, (0, 0, 0, pad_tokens), mode='constant', value=0)
                tmp_text_attention_mask = F.pad(
                    tmp_text_attention_mask, (0, 0, 0, pad_tokens), mode='constant', value=0)
                tmp_expression_token_ids = F.pad(
                    tmp_expression_token_ids, (0, pad_tokens), mode='constant', value=0)
                tmp_gesture_token_ids = F.pad(
                    tmp_gesture_token_ids, (0, pad_tokens), mode='constant', value=0)
                tmp_hands_token_ids = F.pad(
                    tmp_hands_token_ids, (0, pad_tokens), mode='constant', value=0)

            person_data_list.append(
                PersonTokenData(
                    text_input_ids=tmp_text_input_ids,
                    text_attention_mask=tmp_text_attention_mask,
                    audio=tmp_audio,
                    expression_token_ids=tmp_expression_token_ids,
                    gesture_token_ids=tmp_gesture_token_ids,
                    hands_token_ids=tmp_hands_token_ids,
                )
            )

        return person_data_list


def check_partner_data(person_data_self: PersonTokenData, person_data_partner: PersonTokenData):
    for k in person_data_self.__dataclass_fields__:
        v1 = getattr(person_data_self, k)
        v2 = getattr(person_data_partner, k)
        if v1.shape != v2.shape:
            raise ValueError(
                f"shape of {k} is not the same: {v1.shape} != {v2.shape}")
    return
