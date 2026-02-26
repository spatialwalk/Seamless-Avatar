import torch
import numpy as np
import librosa
from configs import SAMPLE_RATE, SMPLX_PARAMS_STATS_SAVE_PATH
from scipy.signal import savgol_filter
from icecream import ic
import os
from configs.joint_masks import joint_mask_upper, joint_mask_arms
from utils.util_transform import sixd2aa, aa2sixd


@torch.no_grad()
def save_motion_dict_to_npz(
    data,
    save_npz_path=None,
    apply_smoothing=False
):
    """
    Inverse operation of load_motion_tensor_from_npz.
    Saves the motion tensor dict back to an .npz file.
    """
    applied_mask = joint_mask_upper
    n_frame = data['body_pose'].shape[0]
    device = data['body_pose'].device

    body_pose_full = torch.zeros((n_frame, 21, 6)).to(device)
    body_pose_full[:, applied_mask] = data['body_pose'].reshape(n_frame, 13, 6)
    data['body_pose'] = body_pose_full

    data['left_hand_pose'] = data['left_hand_pose'].reshape(n_frame, 15, 6)
    data['right_hand_pose'] = data['right_hand_pose'].reshape(n_frame, 15, 6)

    # Denormalize
    normalize = True
    if normalize:
        stats = dict(np.load(SMPLX_PARAMS_STATS_SAVE_PATH))
        for k, v in data.items():
            mean = torch.from_numpy(stats[k+'_mean']).to(device)
            std = torch.from_numpy(stats[k+'_std']).to(device) + 1e-5
            data[k] = v * std + mean

    # Convert poses from 6D to axis-angle
    data['jaw_pose'] = sixd2aa(data['jaw_pose'])
    data['body_pose'] = sixd2aa(data['body_pose'], batch=True)
    data['left_hand_pose'] = sixd2aa(data['left_hand_pose'], batch=True)
    data['right_hand_pose'] = sixd2aa(data['right_hand_pose'], batch=True)

    # Convert to numpy
    data = {k: v.detach().cpu().numpy() for k, v in data.items()}

    if apply_smoothing:
        for k, v in data.items():
            data[k] = savgol_filter(v, 7, 2, axis=0)
        ic('Applied smoothing to motion data before saving.')
        exit()

    # 4. Save to .npz
    # Use keyword arguments (**data) so keys become 'expression', 'body_pose', etc. inside the file
    if save_npz_path is not None:
        os.makedirs(os.path.dirname(save_npz_path), exist_ok=True)
        np.savez(save_npz_path, **data)
    return data


def load_motion_dict_from_npz(
    npz_path,
):
    # data.keys(): dict_keys(['expression', 'jaw_pose', 'body_pose', 'left_hand_pose', 'right_hand_pose'])
    data = dict(np.load(npz_path, allow_pickle=True))

    # check data shape
    assert data['jaw_pose'].ndim == 2
    assert data['expression'].ndim == 2
    assert data['body_pose'].ndim == 3
    assert data['right_hand_pose'].ndim == 3
    assert data['left_hand_pose'].ndim == 3

    assert data['jaw_pose'].shape[1] == 3
    assert data['expression'].shape[1] == 50  # extracted by spectre
    assert data['body_pose'].shape[1:] == (21, 3)
    assert data['left_hand_pose'].shape[1:] == (15, 3)
    assert data['right_hand_pose'].shape[1:] == (15, 3)

    assert len(joint_mask_upper) == data['body_pose'].shape[1]
    assert len(joint_mask_arms) == data['body_pose'].shape[1]

    # compute n_frames
    n_frame_list = [
        v.shape[0] for v in data.values()
    ]
    if max(n_frame_list) - min(n_frame_list) > 2:
        # warnings.warn(f"n_frame_list: {n_frame_list}")
        return None
    n_frame = min(n_frame_list)
    data = {
        k: v[:n_frame] for k, v in data.items()
    }

    # preprocess data
    data = {k: torch.from_numpy(v).float() for k, v in data.items()}
    data['body_pose'] = aa2sixd(data['body_pose'], batch=True)
    data['left_hand_pose'] = aa2sixd(data['left_hand_pose'], batch=True)
    data['right_hand_pose'] = aa2sixd(data['right_hand_pose'], batch=True)
    data['jaw_pose'] = aa2sixd(data['jaw_pose'])

    normalize = True
    if normalize:
        stats = dict(np.load(SMPLX_PARAMS_STATS_SAVE_PATH))
        for k, v in data.items():
            data[k] = (v - stats[k+'_mean']) / (stats[k+'_std']+1e-5)

    data['body_pose'] = data['body_pose'][:, joint_mask_upper]

    # convert body pose to 6d representation

    data['body_pose'] = data['body_pose'].reshape(n_frame, 13*6)
    data['left_hand_pose'] = data['left_hand_pose'].reshape(n_frame, 15*6)
    data['right_hand_pose'] = data['right_hand_pose'].reshape(n_frame, 15*6)

    return data


def load_audio_tensor_from_file(audio_path):
    '''
    load data from file, no preprocess, just load data to be torch.tensor
    '''
    audio, sampling_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
    assert audio.ndim == 1, f"audio.ndim: {audio.ndim} != 1"
    audio = torch.from_numpy(audio)

    audio_mean = audio.mean()
    audio_std = audio.std()
    audio = (audio - audio_mean) / (audio_std+1e-5)

    return audio
