import os
import tqdm
import json
import torch
import torch.distributed as dist
from src.data.load_tensor_from_file import load_motion_dict_from_npz, load_audio_tensor_from_file
from configs import SMPLX_NPZ_FOLDER, AUDIOS_FOLDER, DATASET_ROOT_DIR, SAMPLE_RATE, FPS, TMP_DIR
from .util_data import split_data_to_chunks


class MotionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *args,
        **kwargs,

    ):
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:

            tmp_path = os.path.join(
                TMP_DIR, f"data_cache_{kwargs['split']}.pt")
            if dist.get_rank() == 0:
                os.makedirs(TMP_DIR, exist_ok=True)
                data_list = self.load_data_list(*args, **kwargs)
                torch.save(data_list, tmp_path)
            dist.barrier()
            if dist.get_rank() != 0:
                data_list = torch.load(tmp_path, weights_only=False)
            dist.barrier()
            if dist.get_rank() == 0:
                os.remove(tmp_path)
            self.data_list = data_list
        else:
            self.data_list = self.load_data_list(*args, **kwargs)

    def load_data_list(self, split, debug, motion_type, chunksize):

        json_path = os.path.join(DATASET_ROOT_DIR, f'{split}.json')
        paired_file_short_path_list = json.load(open(json_path, 'r'))
        file_short_path_list = [
            item
            for sublist in paired_file_short_path_list
            for item in sublist
        ]
        npz_path_list = [
            os.path.join(SMPLX_NPZ_FOLDER,  item+'.npz')
            for item in file_short_path_list
        ]

        if debug:
            npz_path_list = npz_path_list[:20]

        if chunksize is not None:
            chunksize_audio = chunksize//FPS*SAMPLE_RATE

        self.chunksize = chunksize

        data_list = []
        for npz_path in tqdm.tqdm(npz_path_list, desc=f'load data from disk [{split}|{motion_type}]'):
            label = os.path.dirname(npz_path).split('/')[-1]

            data_dict = load_motion_dict_from_npz(npz_path)
            if data_dict is None:
                continue

            if motion_type == 'expression':
                motion_data = torch.cat(
                    [data_dict['expression'], data_dict['jaw_pose']],
                    dim=1
                )
            elif motion_type == 'gesture':
                motion_data = data_dict['body_pose']
            elif motion_type == 'hands':
                motion_data = torch.cat(
                    [
                        data_dict['left_hand_pose'],
                        data_dict['right_hand_pose']
                    ],
                    dim=1
                )
            else:
                raise ValueError(f'invalid motion_type: {motion_type}')

            audio_path = os.path.join(
                AUDIOS_FOLDER,
                label,
                os.path.basename(npz_path).replace('.npz', '.wav')
            )
            audio_data = load_audio_tensor_from_file(audio_path)
            n_frame_list = [
                motion_data.shape[0],
                audio_data.shape[0]/SAMPLE_RATE*FPS
            ]
            assert max(n_frame_list) - \
                min(n_frame_list) <= 12, f"n_frame_list: {n_frame_list}"

            if chunksize is not None:
                tmp_motion_list = split_data_to_chunks(
                    motion_data, chunksize)

                if audio_data is not None:
                    tmp_audio_list = split_data_to_chunks(
                        audio_data, chunksize_audio)
                else:
                    tmp_audio_list = [None]*len(tmp_motion_list)

                n_chunk_list = [
                    len(tmp_motion_list),
                    len(tmp_audio_list),
                ]

                assert max(n_chunk_list) - \
                    min(n_chunk_list) == 0,  f"n_chunk_list: {n_chunk_list}"
                n_chunk = min(n_chunk_list)
                data_list.extend(
                    zip(
                        tmp_motion_list[:n_chunk],
                        tmp_audio_list[:n_chunk],
                    )
                )
            else:
                file_rel_path = os.path.relpath(npz_path, SMPLX_NPZ_FOLDER)
                data_list.append(
                    (motion_data, audio_data, file_rel_path))

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        tmp_data = self.data_list[index]

        motion = tmp_data[0]
        audio = tmp_data[1]
        file_rel_path = tmp_data[2] if len(tmp_data) == 3 else None

        data_dict = {'motion': motion}

        if audio is not None:
            data_dict['audio'] = audio

        if file_rel_path is not None:
            data_dict['file_rel_path'] = file_rel_path

        return data_dict
