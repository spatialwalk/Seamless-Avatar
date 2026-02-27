from pathlib import Path
import os
from icecream import ic, install
from src.renderer.smplx_model import render_smplx_to_video
from configs import SMPLX_NPZ_FOLDER, AUDIOS_FOLDER
from utils.media import combine_video_and_audio, stitch_videos


install()


def vis_infer(pred_npz_folder, save_video_folder, text_list):

    os.makedirs(save_video_folder, exist_ok=True)

    pred_npz_path_list = list(Path(pred_npz_folder).rglob('*.npz'))
    pred_npz_path_list = [str(f) for f in pred_npz_path_list]

    # infer
    i = 0
    for pred_npz_path in pred_npz_path_list:
        label = os.path.dirname(pred_npz_path).split('/')[-1]
        audio_path = os.path.join(
            AUDIOS_FOLDER,
            label,
            os.path.basename(pred_npz_path).replace('.npz', '.wav'))
        gt_npz_path = os.path.join(
            SMPLX_NPZ_FOLDER, os.path.relpath(pred_npz_path, pred_npz_folder))

        save_video_path = os.path.join(
            save_video_folder, os.path.basename(pred_npz_path).replace('.npz', '.mp4'))
        save_video_path_gt = save_video_path.replace('.mp4', '_gt.mp4')
        save_video_path_pred = save_video_path.replace('.mp4', '_pred.mp4')

        max_frames = 1200

        smplx_model = render_smplx_to_video(
            pred_npz_path,
            save_video_path_pred,
            smplx_model=None,
            max_frames=max_frames,
        )
        smplx_model = render_smplx_to_video(
            gt_npz_path,
            save_video_path_gt,
            smplx_model=smplx_model,
            max_frames=max_frames,
        )
        stitch_videos([
            save_video_path_gt,
            save_video_path_pred,
        ],
            save_video_path,
            text_list=text_list,
        )

        combine_video_and_audio(save_video_path, audio_path, save_video_path)

        os.remove(save_video_path_gt)
        os.remove(save_video_path_pred)

        i += 1

        ic(save_video_path)
        # exit()
        if i > 10:
            break
    return


if __name__ == '__main__':

    pred_npz_folder = './outputs/DiT_holistic_pred_0108_v1'
    save_video_folder = pred_npz_folder + '_videos'

    vis_infer(
        pred_npz_folder, save_video_folder,
        text_list=['GT', 'DiT']
    )
