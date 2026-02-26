import gc
import os
import tqdm
import torch
from icecream import ic, install
from src.data.load_tensor_from_file import save_motion_dict_to_npz
from src.data.motion_dataset import MotionDataset
from src.DiT.dyadic_model import DyadicTalkingHead, DyadicTalkingHeadConfig
from utils.util_infer import load_model
from utils.util_func import get_motion_dim
install()


if __name__ == '__main__':

    pred_npz_folder = f'./outputs/DiT_holistic_pred_0108_v1'

    DEBUG = True
    infer_split = 'test'

    dit_ckpt_path_map = {
        'gesture': f'models/pretrained_models/DiT_gesture.pt',
        # 'expression': f'outputs/DiT_1228_expression/checkpoints/400.pt',
        # 'hands': f'outputs/DiT_0107_hands/checkpoints/340.pt'
    }

    motion_type_list = list(dit_ckpt_path_map.keys())

    test_dataset = {}
    dit_model = {}
    for motion_type in motion_type_list:
        test_dataset[motion_type] = MotionDataset(
            split=infer_split,
            debug=DEBUG,
            motion_type=motion_type,
            chunksize=None,

        )
        tmp_model_config = DyadicTalkingHeadConfig(
            motion_dim=get_motion_dim(motion_type)
        )
        dit_model[motion_type] = DyadicTalkingHead(tmp_model_config).to('cuda')
        dit_model[motion_type] = load_model(
            dit_model[motion_type], dit_ckpt_path_map[motion_type])

    dataset_len_list = [len(test_dataset[motion_type])
                        for motion_type in motion_type_list]
    assert len(set(dataset_len_list)) == 1
    dataset_len = dataset_len_list[0]
    ic(dataset_len)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in tqdm.tqdm(range(dataset_len), desc='Inferring DiT'):

        motion_gt = {}
        motion_pred = {}
        file_rel_path = {}

        n_frames = None

        for motion_type in motion_type_list:
            # get data
            tmp_data_dict = test_dataset[motion_type][i]
            motion_gt[motion_type] = tmp_data_dict['motion']
            file_rel_path[motion_type] = tmp_data_dict['file_rel_path']

            # infer
            motion_pred[motion_type] = dit_model[motion_type].sample(
                tmp_data_dict['audio'],
                num_steps=50
            )[0]

            ic(motion_pred[motion_type].shape)
            ic(motion_gt[motion_type].shape)
            n_frames = motion_gt[motion_type].shape[0]

        assert n_frames is not None

        if 'gesture' not in motion_type_list:
            motion_pred['gesture'] = torch.zeros(
                (n_frames, get_motion_dim('gesture'))).to(device)

        if 'expression' not in motion_type_list:
            motion_pred['expression'] = torch.zeros(
                (n_frames, get_motion_dim('expression'))).to(device)

        if 'hands' not in motion_type_list:
            motion_pred['hands'] = torch.zeros(
                (n_frames, get_motion_dim('hands'))).to(device)

        file_rel_path_list = list(file_rel_path.values())
        assert len(set(file_rel_path_list)) == 1
        file_rel_path = file_rel_path_list[0]

        n_frame_list = [motion_gt[motion_type].shape[0]
                        for motion_type in motion_type_list]
        assert len(set(n_frame_list)) == 1

        motion_dict_pred = {
            'body_pose': motion_pred['gesture'],
            'left_hand_pose': motion_pred['hands'][:, :15*6],
            'right_hand_pose': motion_pred['hands'][:, 15*6:],
            'expression': motion_pred['expression'][:, :50],
            'jaw_pose': motion_pred['expression'][:, 50:],
        }

        save_npz_path_pred = os.path.join(pred_npz_folder, file_rel_path)

        save_motion_dict_to_npz(
            motion_dict_pred, save_npz_path_pred
        )
        ic(save_npz_path_pred)

        torch.cuda.empty_cache()
        gc.collect()
