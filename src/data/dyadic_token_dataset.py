from utils.env_config import *
import os
import tqdm
import json
import torch
from configs import FRAMES_PER_TOKEN
from icecream import ic, install
from src.data.load_tensor_from_file import load_audio_tensor_from_file, load_aligned_text_list
from configs import AUDIOS_FOLDER, DATASET_ROOT_DIR, SAMPLE_RATE, FPS, FRAMES_PER_TOKEN, ASR_FOLDER
from .person_token_data import PersonTokenData, check_partner_data

install()


def _parse_filename(filename):
    vendor_seesion_interaction_prefix = os.path.basename(filename).split('_')[
        0:3]
    return vendor_seesion_interaction_prefix


def _load_personData_from_fileshortName(fileshort_name, motion_token_ids_folder):

    label = os.path.dirname(fileshort_name)

    # audio
    audio_path = os.path.join(
        AUDIOS_FOLDER,
        label+'_loudnorm_16k',
        os.path.basename(fileshort_name)+'.wav'
    )
    audio_data = load_audio_tensor_from_file(audio_path)

    # asr
    asr_path = os.path.join(
        ASR_FOLDER,
        fileshort_name+'.json',
    )

    # motion token ids
    motion_token_json_path = os.path.join(
        motion_token_ids_folder,
        fileshort_name+'_smplx.json'
    )
    motion_token_ids = json.load(
        open(motion_token_json_path, 'r'))
    expression_token_ids = torch.tensor(motion_token_ids['expression'])
    gesture_token_ids = torch.tensor(motion_token_ids['gesture'])
    hands_token_ids = torch.tensor(motion_token_ids['hands'])

    text_input_ids, text_attention_mask = load_aligned_text_list(
        asr_path, n_tokens=expression_token_ids.shape[0])

    return PersonTokenData(
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask,
        audio=audio_data,
        expression_token_ids=expression_token_ids,
        gesture_token_ids=gesture_token_ids,
        hands_token_ids=hands_token_ids,
    )


def load_dyadic_token_data_from_disk(
        split, debug, motion_token_ids_folder, chunksize):
    """
    Returns:
        list: [(person_0_data, person_1_data), ...] 所有数据对
    """
    json_path = os.path.join(DATASET_ROOT_DIR, f'{split}.json')
    paired_file_short_path_list = json.load(open(json_path, 'r'))
    if chunksize is not None:
        chunksize_audio = chunksize//FPS*SAMPLE_RATE
        chunksize_token = chunksize // FRAMES_PER_TOKEN
        ic(chunksize, chunksize_audio, chunksize_token)

    if debug:
        paired_file_short_path_list = paired_file_short_path_list[:20]

    data_list = []
    for paired_file_short_path in tqdm.tqdm(paired_file_short_path_list, desc=f'load data from disk [{split}|DyadicTokenDataset]'):
        file_short_path_0, file_short_path_1 = paired_file_short_path

        assert _parse_filename(
            file_short_path_0) == _parse_filename(file_short_path_1)

        paired_file_short_path_0 = paired_file_short_path[0]
        paired_file_short_path_1 = paired_file_short_path[1]

        person_data_0 = _load_personData_from_fileshortName(
            paired_file_short_path_0, motion_token_ids_folder)
        person_data_1 = _load_personData_from_fileshortName(
            paired_file_short_path_1, motion_token_ids_folder)

        check_partner_data(person_data_0, person_data_1)
        if chunksize is not None:
            person_0_data_list = person_data_0.split_to_chunks(
                chunksize_token, chunksize_audio)
            person_1_data_list = person_data_1.split_to_chunks(
                chunksize_token, chunksize_audio)
            assert len(person_0_data_list) == len(person_1_data_list)

            data_list.extend(list(zip(person_0_data_list, person_1_data_list)))
        else:
            data_list.append((person_data_0, person_data_1,
                             paired_file_short_path_0, paired_file_short_path_1))

    return data_list


class DyadicTokenDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        debug,
        motion_token_ids_folder,
        chunksize=None
    ):

        self.data_list = load_dyadic_token_data_from_disk(
            split, debug, motion_token_ids_folder, chunksize)
        self.chunksize = chunksize

    def __len__(self):
        return len(self.data_list)*2  # don't forget this *2

    def __getitem__(self, index):
        if index >= len(self.data_list):
            index = index - len(self.data_list)
            exchange_roles = True
        else:
            exchange_roles = False

        tmp_data = self.data_list[index]
        person_0_data = tmp_data[0]
        person_1_data = tmp_data[1]

        if exchange_roles:
            person_0_data, person_1_data = person_1_data, person_0_data

        data_dict = {
            'user': person_0_data.to_dict(),
            'agent': person_1_data.to_dict(),
        }

        if self.chunksize is None:
            data_dict['user']['file_rel_path'] = tmp_data[2]
            data_dict['agent']['file_rel_path'] = tmp_data[3]

        return data_dict
