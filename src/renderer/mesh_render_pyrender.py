import trimesh
import pyrender
import numpy as np
import imageio
import tqdm
import os


def light_direction_to_pose_pyrender(direction):
    direction = np.asarray(direction, dtype=np.float32)
    direction /= np.linalg.norm(direction)
    # 光照方向 = direction（光从该方向来）
    # 局部 -Z 应 = direction → 局部 Z = -direction
    z = -direction
    x = np.cross([0, 0, 1], z)
    if np.linalg.norm(x) < 1e-6:  # direction 接近 Z 轴
        x = np.cross([1, 0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    pose = np.eye(4)
    pose[:3, :3] = R
    return pose


def create_scene_with_mesh_pyrender(vertices, faces, uniform_color, pose_camera, light_pose, bg_color):
    trimesh_mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_colors=uniform_color)
    # mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=True)

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[uniform_color[0]/255.0,
                         uniform_color[1]/255.0, uniform_color[2]/255.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True  # 单面渲染
    )
    mesh = pyrender.Mesh.from_trimesh(
        trimesh_mesh,
        material=material,
        smooth=True)

    scene = pyrender.Scene(bg_color=bg_color)
    scene.add(mesh)
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    scene.add(camera, pose=pose_camera)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    scene.add(light, pose=light_pose)
    return scene


class MeshRendererPyrender:
    def __init__(self, fig_resolution, bg_color=[1.0, 1.0, 1.0]):
        self.renderer = pyrender.OffscreenRenderer(*fig_resolution)
        self.background_color = bg_color

    def render_to_img(self, vertices, faces, color, camera, light_dir, save_img_path=None):
        light_pose = light_direction_to_pose_pyrender(light_dir)
        scene = create_scene_with_mesh_pyrender(
            vertices, faces, color, camera, light_pose, self.background_color)
        fig, _ = self.renderer.render(scene)

        if save_img_path is not None:
            imageio.imwrite(save_img_path, fig)
        return fig

    def render_to_video(
        self, vertices, faces, color, camera, light_dir,
        save_video_path=None, fps=30
    ):
        assert vertices.ndim == 3 and vertices.shape[2] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3

        img_list = []
        for i in tqdm.trange(vertices.shape[0], desc="Rendering[Pyrender]"):
            fig = self.render_to_img(
                vertices[i], faces, color, camera, light_dir)
            img_list.append(fig)

        if save_video_path is not None:
            imageio.mimsave(save_video_path, img_list, fps=fps)

        return img_list
