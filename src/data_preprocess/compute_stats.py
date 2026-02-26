import os
from pydoc import describe
from configs import DATASET_ROOT_DIR, SMPLX_PARAMS_STATS_SAVE_PATH
from icecream import ic
import numpy as np
import tqdm
from pathlib import Path
from collections import defaultdict
from tqdm.contrib.concurrent import process_map
from utils.util_transform import aa2sixd
import torch
import random


def load_smplx_params_from_npz(npz_file):
    data_dict = dict(np.load(npz_file))
    return data_dict


def compute_smplx_params_stats():
    # if os.path.exists(SMPLX_PARAMS_STATS_SAVE_PATH):
    #     print(f'{SMPLX_PARAMS_STATS_SAVE_PATH} already exists')
    #     return

    data_path = Path(DATASET_ROOT_DIR) / 'smplx_npz_annos'
    npz_file_list = list(Path(data_path).rglob('*_smplx.npz'))
    npz_file_list = [str(f) for f in npz_file_list]
    random.shuffle(npz_file_list)
    npz_file_list = npz_file_list[:1000]
    
    smplx_params_list = process_map(
        load_smplx_params_from_npz, npz_file_list, desc='Loading smplx params')  # list of dict

    smplx_params_stats = {}
    for k, v in smplx_params_list[0].items():
        all_data = np.concatenate([d[k] for d in smplx_params_list], axis=0)

        if 'pose' in k:
            assert all_data.shape[-1] == 3
            all_data = torch.from_numpy(all_data)
            if all_data.ndim == 2:
                all_data = aa2sixd(all_data)
            elif all_data.ndim == 3:
                all_data = aa2sixd(all_data, batch=True)
            else:
                raise ValueError(f"Invalid data shape: {all_data.shape}")
            all_data = all_data.numpy()

        smplx_params_stats[k+'_mean'] = all_data.mean(axis=0)
        smplx_params_stats[k+'_std'] = all_data.std(axis=0)

    np.savez(SMPLX_PARAMS_STATS_SAVE_PATH, **smplx_params_stats)
    return


def inspect_smplx_params_stats():
    stats = dict(np.load(SMPLX_PARAMS_STATS_SAVE_PATH))
    for k, v in stats.items():
        # print('-'*100)
        print(k, v.shape)
        print(v)


if __name__ == '__main__':

    # compute_smplx_params_stats()
    inspect_smplx_params_stats()
