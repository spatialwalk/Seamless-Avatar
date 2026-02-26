import os
from icecream import ic
from src.renderer.smplx_model import render_smplx_to_video
from utils.media import combine_video_and_audio

if __name__ == "__main__":
    from pathlib import Path
    import random

    root_dir = './datasets/seamless_smplx_dataset/'
    save_video_folder = 'outputs/smplx_videos'

    smplx_npz_folder = os.path.join(root_dir, 'smplx_npz_annos')
    audio_folder = os.path.join(root_dir, 'audios_loudnorm_16k')

    npz_path_list = list(Path(smplx_npz_folder).rglob('*_smplx.npz'))
    npz_path_list = [str(f) for f in npz_path_list]
    # random.shuffle(npz_path_list)

    os.makedirs(save_video_folder, exist_ok=True)

    for npz_path in npz_path_list:
        label = os.path.dirname(npz_path).split('/')[-1]

        audio_path = os.path.join(
            audio_folder,
            label+'_loudnorm_16k',
            os.path.basename(npz_path).replace('_smplx.npz', '.wav'))

        save_video_path = os.path.join(
            save_video_folder, os.path.basename(npz_path).replace('.npz', '.mp4'))
        smplx_model = render_smplx_to_video(
            npz_path,
            save_video_path,
            smplx_model=None,
        )
        combine_video_and_audio(save_video_path, audio_path, save_video_path)
        ic(save_video_path)
        break
