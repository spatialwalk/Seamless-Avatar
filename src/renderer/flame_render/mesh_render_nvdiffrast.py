"""
使用 MeshRenderer 类的完整示例

展示如何使用渲染器渲染 FLAME mesh 并保存为视频
"""
import os
import torch
import imageio
from src.metrics.flame_mesh import FlameMesh
from src.renderer import MeshRenderer
import tqdm
from icecream import ic
from utils.media import combine_video_and_audio
import numpy as np


def save_img_tensor_list_to_video(all_imgs, video_path):
    with imageio.get_writer(video_path, fps=25, codec='libx264', quality=8) as writer:
        for img_tensor in all_imgs:
            img_np = img_tensor.cpu().numpy()
            writer.append_data(img_np)
    return


def save_img_tensor_to_file(img, img_path):
    imageio.imwrite(img_path, img.cpu())
    return


class MeshRenderNvdiffrast:
    def __init__(self):
        # 初始化
        self.flame_mesh = FlameMesh()
        self.renderer = MeshRenderer(
            image_width=512, image_height=512, device='cuda')
        self.faces = self.flame_mesh.flame.faces_tensor

    def _render(self, data_dict, video_path, ignore_global_rot, audio_path):
        verts, landmarks2d, landmarks3d = self.flame_mesh.get_vertices(
            data_dict, ignore_global_rot=ignore_global_rot)

        all_imgs = self.renderer.render_img_list_batchwise(verts, self.faces)

        # 保存为视频
        save_img_tensor_list_to_video(all_imgs, video_path)

        if audio_path is not None:

            combine_video_and_audio(video_path, audio_path, video_path)

        return video_path

    def render_to_video(self, flame_params, video_path, ignore_global_rot=False,  audio_path=None):
        if isinstance(flame_params, str) and flame_params.endswith('.npz'):
            data_dict = dict(np.load(flame_params, allow_pickle=True))

        elif isinstance(flame_params, dict):
            data_dict = flame_params
        else:
            raise ValueError(f"Invalid flame_params: {flame_params}")

        return self._render(data_dict, video_path, ignore_global_rot, audio_path)
