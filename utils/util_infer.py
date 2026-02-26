
import torch
from configs import SAMPLE_RATE, FPS
import librosa
from icecream import ic
import numpy as np
import json
import os
import math


def compute_speech_ratio(audio_path: str, top_db: int = 20) -> float:
    """
    计算整段音频中“有声(非静音)”部分所占比例。

    参数
    -------
    audio_path : str
        wav 文件路径
    top_db : int
        静音界定阈值（dB）。数值越小，判定越“严格”，
        > 20 dB 一般能覆盖普通环境；可按数据情况微调。

    返回
    -------
    ratio : float
        0–1 之间，说话(或任何有声)所占比例
    """
    # 1. 读入波形
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # 2. 找到所有非静音区间 [start, end)
    # librosa 先计算短时能量，再把低于 (max_energy - top_db) 的帧过滤
    intervals = librosa.effects.split(y, top_db=top_db)

    # 3. 统计“有声”样本数
    speech_samples = sum(end - start for start, end in intervals)

    # 4. 计算比例
    return speech_samples / len(y)


def list_npz(folder: str):
    lst = [os.path.join(folder, f)
           for f in os.listdir(folder) if f.endswith('.npz')]
    lst.sort()
    return lst


def chunk_range(total_samples: int, total_chunks: int, chunk_id: int):
    """返回 [start_idx, end_idx)"""
    per_chunk = math.ceil(total_samples / total_chunks)
    start = chunk_id * per_chunk
    end = min(start + per_chunk, total_samples)
    return start, end


def get_partner_file(file_path):
    postfix = file_path.split('.')[-1]

    file_base = file_path.split('.'+postfix)[0]

    if file_base.endswith('_speaker1'):
        partner_file_base = file_base.replace('_speaker1', '_speaker2')
    elif file_base.endswith('_speaker2'):
        partner_file_base = file_base.replace('_speaker2', '_speaker1')
    else:
        raise ValueError(f'Invalid file path: {file_path}')
    return f'{partner_file_base}.{postfix}'


def load_model(model, ckpt_path, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    ckpt = torch.load(ckpt_path)
    state = ckpt['model']

    # 关键：跳过旧 ckpt 里的 vqvae 部分
    state = {k: v for k, v in state.items(
    ) if not k.startswith("motion_vqvae.")}

    incompatible = model.load_state_dict(state, strict=True)
    # 你想更严格也行：过滤完再 strict=True；但模型有其他改动时会报 missing keys
    print("missing_keys:", incompatible.missing_keys)
    print("unexpected_keys:", incompatible.unexpected_keys)

    model.eval()

    if verbose:
        ic(ckpt['loss_dict'])
        ic(ckpt['epoch'])
        
    return model
