from .emage_evaltools.metric import FGD, BC, LVDFace, L1div, MSEFace
import numpy as np
import torch
from icecream import ic, install
from .emage_evaltools import rotation_conversions as rc
import os
from pathlib import Path
import tqdm
from configs.joint_masks import joint_mask_lower
from configs import DATASET_ROOT_DIR, SMPLX_NPZ_FOLDER, AUDIOS_FOLDER, SAMPLE_RATE
import json
import pandas as pd
import smplx
from termcolor import colored, cprint
import librosa
install()

# -------------- file_short_path_list -------------------
json_path = os.path.join(DATASET_ROOT_DIR, f'test.json')
paired_file_short_path_list = json.load(open(json_path, 'r'))
file_short_path_list = [
    item
    for sublist in paired_file_short_path_list
    for item in sublist
]

# -------------- initialize evaluators -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
download_path = os.path.join(os.path.dirname(__file__), 'emage_evaltools')

fgd_evaluator = FGD(download_path=download_path)
bc_evaluator = BC(download_path=download_path, sigma=0.3, order=7,threshold=0.02)
l1div_evaluator = L1div()
lvd_evaluator = LVDFace()
mse_evaluator = MSEFace()

smplx_model = smplx.create(
    os.path.join(download_path, 'smplx_models'),
    model_type='smplx',
    gender='NEUTRAL_2020',
    use_face_contour=False,
    num_betas=300,
    num_expression_coeffs=100,
    ext='npz',
    use_pca=False,
).eval()


def get_motion_rep_numpy(poses_np, pose_fps=30, device="cuda", expressions=None, expression_only=False, betas=None):
    # motion["poses"] is expected to be numpy array of shape (n, 165)
    # (n, 55*3), axis-angle for 55 joints
    global smplx_model
    smplx_model = smplx_model.to(device)
    n = poses_np.shape[0]

    # Convert numpy to torch tensor for SMPL-X forward pass
    poses_ts = torch.from_numpy(poses_np).float().to(
        device).unsqueeze(0)  # (1, n, 165)
    poses_ts_reshaped = poses_ts.reshape(-1, 165)  # (n, 165)
    betas = torch.zeros(n, 300, device=device) if betas is None else torch.from_numpy(
        betas).to(device).unsqueeze(0).repeat(n, 1)
    if expressions is not None and expression_only:
        # print("xx")
        expressions = torch.from_numpy(expressions).float().to(device)
        output = smplx_model(
            betas=betas,
            transl=torch.zeros(n, 3, device=device),
            expression=expressions,
            jaw_pose=poses_ts_reshaped[:, 22 * 3:23 * 3],
            global_orient=torch.zeros(n, 3, device=device),
            body_pose=torch.zeros(n, 21*3, device=device),
            left_hand_pose=torch.zeros(n, 15*3, device=device),
            right_hand_pose=torch.zeros(n, 15*3, device=device),
            return_joints=True,
            leye_pose=torch.zeros(n, 3, device=device),
            reye_pose=torch.zeros(n, 3, device=device),
        )
        # joints = output["vertices"].detach().cpu().numpy().reshape(n, -1)
        joints = output["vertices"].reshape(n, -1)
        return {"vertices": joints}

    # Run smplx model to get joints
    output = smplx_model(
        betas=betas,
        transl=torch.zeros(n, 3, device=device),
        expression=torch.zeros(n, 100, device=device),
        jaw_pose=torch.zeros(n, 3, device=device),
        global_orient=torch.zeros(n, 3, device=device),
        body_pose=poses_ts_reshaped[:, 3:21 * 3 + 3],
        left_hand_pose=poses_ts_reshaped[:, 25 * 3:40 * 3],
        right_hand_pose=poses_ts_reshaped[:, 40 * 3:55 * 3],
        return_joints=True,
        leye_pose=torch.zeros(n, 3, device=device),
        reye_pose=torch.zeros(n, 3, device=device),
    )
    joints = output["joints"].reshape(n, 127, 3)[:, :55, :]

    dt = 1 / pose_fps
    # Compute linear velocity
    init_vel = (joints[1:2] - joints[0:1]) / dt
    middle_vel = (joints[2:] - joints[:-2]) / (2 * dt)
    final_vel = (joints[-1:] - joints[-2:-1]) / dt
    vel = torch.cat([init_vel, middle_vel, final_vel], axis=0)

    position = joints

    # Compute rotation 6D from axis-angle
    poses_ts_reshaped_aa = poses_ts.reshape(1, n, 55, 3)
    rot_matrices = rc.axis_angle_to_matrix(poses_ts_reshaped_aa)[
        0]  # (n, 55, 3, 3)
    rot6d = rc.matrix_to_rotation_6d(
        rot_matrices).reshape(n, 55, 6)  # .cpu().numpy()

    # Compute angular velocity
    poses_np = torch.from_numpy(poses_np).to(device)
    init_vel_ang = (poses_np[1:2] - poses_np[0:1]) / dt
    middle_vel_ang = (poses_np[2:] - poses_np[:-2]) / (2 * dt)
    final_vel_ang = (poses_np[-1:] - poses_np[-2:-1]) / dt
    angular_velocity = torch.cat(
        [init_vel_ang, middle_vel_ang, final_vel_ang], axis=0).reshape(n, 55, 3)

    # rep15d: position(55*3), vel(55*3), rot6d(55*6), angular_velocity(55*3) => total 55*(3+3+6+3)=55*15
    rep15d = torch.cat(
        [position, vel, rot6d, angular_velocity], axis=2).reshape(n, 55 * 15)

    return {
        "position": position,
        "velocity": vel,
        "rotation": rot6d,
        "axis_angle": poses_np,
        "angular_velocity": angular_velocity,
        "rep15d": rep15d,
    }


def load_npz_for_metric(npz_path):

    data = dict(np.load(npz_path))
    body_pose = data['body_pose']
    n_frame = body_pose.shape[0]

    body_pose[:, joint_mask_lower] = 0  # set lower body pose to 0

    left_hand_pose = data['left_hand_pose']
    right_hand_pose = data['right_hand_pose']
    jaw_pose = data['jaw_pose'][:, np.newaxis, :]  # (n_frame, 1, 3)

    pelvis_pose = np.zeros((n_frame, 1, 3))
    left_eyepose = np.zeros((n_frame, 1, 3))
    right_eyepose = np.zeros((n_frame, 1, 3))

    expression = data['expression']
    pad_zeros = np.zeros((expression.shape[0], 100 - expression.shape[1]))
    padded_expression = np.concatenate([expression, pad_zeros], axis=1)

    motion = np.concatenate(
        [
            pelvis_pose,
            body_pose,
            jaw_pose,
            left_eyepose,
            right_eyepose,
            left_hand_pose,
            right_hand_pose
        ], axis=1
    )  # (n_frame, 55, 3)
    assert motion.shape == (n_frame, 55, 3)
    return motion, padded_expression


def compute_metrics(pred_folder, gt_folder, file_short_path_list,  debug):

    # -------------- reset evaluators -------------------
    fgd_evaluator.reset()
    bc_evaluator.reset()
    l1div_evaluator.reset()
    lvd_evaluator.reset()
    mse_evaluator.reset()

    # -------------- load data -------------------
    pred_npz_path_list = [
        os.path.join(pred_folder,  item+'_smplx.npz')
        for item in file_short_path_list
    ]
    if debug:
        pred_npz_path_list = pred_npz_path_list[:5]

    # -------------- compute metrics -------------------
    ii = 0
    for pred_npz_path in tqdm.tqdm(pred_npz_path_list, desc=f'Calculating Metrics for {os.path.basename(pred_folder)}'):
        # ii+=1
        # if ii !=2:
        #     continue
        gt_npz_path = pred_npz_path.replace(pred_folder, gt_folder)
        label = os.path.dirname(gt_npz_path).split('/')[-1]
        audio_path = os.path.join(
            AUDIOS_FOLDER,
            label+'_loudnorm_16k',
            os.path.basename(gt_npz_path).replace('_smplx.npz', '.wav')
        )

        motion_pred, expressions_pred = load_npz_for_metric(pred_npz_path)
        motion_gt, expressions_gt = load_npz_for_metric(gt_npz_path)

        # -------------- align frames -------------------
        n_frame_list = [motion_pred.shape[0], motion_gt.shape[0]]
        assert max(n_frame_list) - \
            min(n_frame_list) <= 4, f"n_frame_list: {n_frame_list}"
        n_frame = min(n_frame_list)
        motion_pred = motion_pred[:n_frame]
        motion_gt = motion_gt[:n_frame]
        expressions_pred = expressions_pred[:n_frame]
        expressions_gt = expressions_gt[:n_frame]

        # -------------- compute lvd, l1div, mse -------------------
        motion_position_pred = get_motion_rep_numpy(motion_pred, device=device)[
            "position"]  # n_frame x 55 x 3
        motion_position_gt = get_motion_rep_numpy(motion_gt, device=device)[
            "position"]  # n_frame x 55 x 3
        l1div_evaluator.compute(motion_position_pred)

        face_position_pred = get_motion_rep_numpy(
            motion_pred, device=device, expressions=expressions_pred, expression_only=True)["vertices"]  # t -1
        face_position_gt = get_motion_rep_numpy(
            motion_gt, device=device, expressions=expressions_gt, expression_only=True)["vertices"]
        lvd_evaluator.compute(face_position_pred, face_position_gt)
        mse_evaluator.compute(face_position_pred, face_position_gt)

        # -------------- compute bc -------------------
        # bc_evaluator.compute(motion_position_pred.cpu().numpy(), audio_path, pose_fps=30)
        # bc_evaluator.compute(motion_pred, audio_path, pose_fps=30)

        # print(bc_evaluator.avg())
        # exit()

        # -------------- compute fgd -------------------
        motion_pred = torch.from_numpy(motion_pred).to(device).unsqueeze(0)
        motion_gt = torch.from_numpy(motion_gt).to(device).unsqueeze(0)

        motion_pred = rc.axis_angle_to_rotation_6d(
            motion_pred
        ).reshape(1, n_frame, 55*6)
        motion_gt = rc.axis_angle_to_rotation_6d(
            motion_gt
        ).reshape(1, n_frame, 55*6)

        fgd_evaluator.update(motion_pred.float(), motion_gt.float())

    return {
        'FGD': fgd_evaluator.compute().item(),
        'L1div': l1div_evaluator.avg(),
        'LVD': lvd_evaluator.avg(),
        'MSE':  mse_evaluator.avg(),
        # 'BC': 0,
    }


def print_metric_table(metric_collection, metric_name_direction_map):

    df = pd.DataFrame(metric_collection).T

    # 表格总结
    print('\n' + '='*100)
    print('Summary Table')
    print('='*100)

    # 打印表头
    # 计算 pred_folder 的最大长度
    max_folder_len = max(len(str(idx)) for idx in df.index)
    # 加一些 padding，至少保留一定宽度
    folder_col_width = max(max_folder_len + 2, 30)  # 至少30个字符
    col_width = 12

    # 打印表头
    header = f"{'':>{folder_col_width}s}"  # 动态宽度

    for col in df.columns:
        header += f"{col:>{col_width-2}s}"
        if metric_name_direction_map[col] == 'lower':
            header += " ↓"
        else:
            header += " ↑"

    print(colored(header, 'cyan', attrs=['bold']))
    print('-'*100)

    # 对每一列找出最优和次优值
    best_values = {}

    for col in df.columns:
        sorted_values = df[col].sort_values()
        if metric_name_direction_map[col] == 'lower':
            best_values[col] = sorted_values.iloc[0]
        else:
            best_values[col] = sorted_values.iloc[-1]

    # 打印每一行
    for idx, row in df.iterrows():
        # 打印 pred_folder（缩短路径）
        folder_name = str(idx)
        line = f"{folder_name:{folder_col_width}s}"

        # 打印每个指标
        for col in df.columns:
            value = row[col]
            formatted_value = f"{value:.2e}"

            # 判断是否为最优或次优
            if value == best_values[col]:
                # colored_value = colored(f"{formatted_value:>12s}", 'yellow')
                colored_value = f"{formatted_value:>{col_width}s}"
            else:
                colored_value = f"{formatted_value:>{col_width}s}"

            line += colored_value

        print(line)

    print('='*100)


if __name__ == "__main__":

    # -------------- load data -------------------
    dit_pred_folder = 'outputs/DiT_holistic_pred_0108_v1'
    vqvae_recon_folder = 'outputs/vqvae_holistic_pred_0107'

    gt_folder = SMPLX_NPZ_FOLDER
    pred_folder_list = [
        # gt_folder,
        dit_pred_folder,
        vqvae_recon_folder
    ]

    metric_name_direction_map = {
        'FGD': 'lower',
        'L1div': 'upper',
        'LVD': 'lower',
        'MSE': 'lower',
        'BC': 'lower',
    }

    debug = False
    metric_collection = {}
    for pred_folder in pred_folder_list:

        metrics = compute_metrics(
            pred_folder, gt_folder, file_short_path_list, debug)

        metric_collection[
            os.path.basename(pred_folder)
        ] = metrics

    print_metric_table(metric_collection, metric_name_direction_map)
