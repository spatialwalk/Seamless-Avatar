# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


from typing import Tuple, Literal
import torch
import torch.nn.functional as F
import math
import numpy as np
from scipy.spatial.transform import Rotation


def align_cameras_to_axes(
    R: torch.Tensor,
    T: torch.Tensor,
    target_convention: Literal["opengl", "opencv"] = None,
):
    """align the averaged axes of cameras with the world axes.

    Args:
        R: rotation matrix (N, 3, 3)
        T: translation vector (N, 3)
    """
    # The column vectors of R are the basis vectors of each camera.
    # We construct new bases by taking the mean directions of axes, then use Gram-Schmidt
    # process to make them orthonormal
    bases_c2w = gram_schmidt_orthogonalization(R.mean(0))
    if target_convention == "opengl":
        bases_c2w[:, [1, 2]] *= -1  # flip y and z axes
    elif target_convention == "opencv":
        pass
    bases_w2c = bases_c2w.t()

    # convert the camera poses into the new coordinate system
    R = bases_w2c[None, ...] @ R
    T = bases_w2c[None, ...] @ T
    return R, T


def convert_camera_convention(camera_convention_conversion: str, R: torch.Tensor, K: torch.Tensor, H: int, W: int):
    if camera_convention_conversion is not None:
        if camera_convention_conversion == "opencv->opengl":
            R[:, :3, [1, 2]] *= -1
            # R[:, :3, 2] *= -1  # used by jiangp
            # flip y of the principal point
            K[..., 1, 2] = H - K[..., 1, 2]
        elif camera_convention_conversion == "opencv->pytorch3d":
            R[:, :3, [0, 1]] *= -1
            # flip x and y of the principal point
            K[..., 0, 2] = W - K[..., 0, 2]
            K[..., 1, 2] = H - K[..., 1, 2]
        elif camera_convention_conversion == "opengl->pytorch3d":
            R[:, :3, [0, 2]] *= -1
            # flip x of the principal point
            K[..., 0, 2] = W - K[..., 0, 2]
        else:
            raise ValueError(
                f"Unknown camera coordinate conversion: {camera_convention_conversion}."
            )
    return R, K


def gram_schmidt_orthogonalization(M: torch.tensor):
    """conducting Gram-Schmidt process to transform column vectors into orthogonal bases

    Args:
        M: An matrix (num_rows, num_cols)
    Return:
        M: An matrix with orthonormal column vectors (num_rows, num_cols)
    """
    num_rows, num_cols = M.shape
    for c in range(1, num_cols):
        M[:, [c - 1, c]] = F.normalize(M[:, [c - 1, c]], p=2, dim=0)
        M[:, [c]] -= M[:, :c] @ (M[:, :c].T @ M[:, [c]])

    M[:, -1] = F.normalize(M[:, -1], p=2, dim=0)
    return M


def projection_from_intrinsics(K: np.ndarray, image_size: Tuple[int], near: float=0.01, far:float=10, flip_y: bool=False, z_sign=-1):
    """
    Transform points from camera space (x: right, y: up, z: out) to clip space (x: right, y: down, z: in)
    Args:
        K: Intrinsic matrix, (N, 3, 3)
            K = [[
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1],
                ]
            ]
        image_size: (height, width)
    Output:
        proj = [[
                [2*fx/w, 0.0,     (w - 2*cx)/w,             0.0                     ],
                [0.0,    2*fy/h, (h - 2*cy)/h,             0.0                     ],
                [0.0,    0.0,     z_sign*(far+near) / (far-near), -2*far*near / (far-near)],
                [0.0,    0.0,     z_sign,                     0.0                     ]
            ]
        ]
    """

    B = K.shape[0]
    h, w = image_size

    if K.shape[-2:] == (3, 3):
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
    elif K.shape[-1] == 4:
        # fx, fy, cx, cy = K[..., [0, 1, 2, 3]].split(1, dim=-1)
        fx = K[..., [0]]
        fy = K[..., [1]]
        cx = K[..., [2]]
        cy = K[..., [3]]
    else:
        raise ValueError(f"Expected K to be (N, 3, 3) or (N, 4) but got: {K.shape}")

    proj = np.zeros([B, 4, 4])
    proj[:, 0, 0]  = fx * 2 / w 
    proj[:, 1, 1]  = fy * 2 / h
    proj[:, 0, 2]  = (w - 2 * cx) / w
    proj[:, 1, 2]  = (h - 2 * cy) / h
    proj[:, 2, 2]  = z_sign * (far+near) / (far-near)
    proj[:, 2, 3]  = -2*far*near / (far-near)
    proj[:, 3, 2]  = z_sign

    if flip_y:
        proj[:, 1, 1] *= -1
    return proj


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, znear=1e-8, zfar=10, convention: Literal["opengl", "opencv"]="opengl"):
        self.image_width = W
        self.image_height = H
        self.radius_default = r
        self.fovy_default = fovy
        self.znear = znear
        self.zfar = zfar
        self.convention = convention

        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.reset()
    
    def reset(self):
        """ The internal state of the camera is based on the OpenGL convention, but 
            properties are converted to the target convention when queried.
        """
        self.rot = Rotation.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # OpenGL convention
        self.look_at = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.radius = self.radius_default  # camera distance from center
        self.fovy = self.fovy_default
        if self.convention == "opencv":
            self.z_sign = 1
            self.y_sign = 1
        elif self.convention == "opengl":
            self.z_sign = -1
            self.y_sign = -1
        else:
            raise ValueError(f"Unknown convention: {self.convention}")

    @property
    def fovx(self):
        return self.fovy / self.image_height * self.image_width
    
    @property
    def intrinsics(self):
        focal = self.image_height / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.image_width // 2, self.image_height // 2])
    
    @property
    def projection_matrix(self):
        return projection_from_intrinsics(self.intrinsics[None], (self.image_height, self.image_width), self.znear, self.zfar, z_sign=self.z_sign)[0]
    
    @property
    def world_view_transform(self):
        return np.linalg.inv(self.pose)  # world2cam

    @property
    def full_proj_transform(self):
        return self.projection_matrix @ self.world_view_transform

    @property
    def pose(self):
        # first move camera to radius
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] += self.radius

        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        pose = rot @ pose

        # translate
        pose[:3, 3] -= self.look_at

        if self.convention == "opencv":
            pose[:, [1, 2]] *= -1
        elif self.convention == "opengl":
            pass
        else:
            raise ValueError(f"Unknown convention: {self.convention}")
        return pose

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.3 * dx)
        rotvec_y = side * np.radians(-0.3 * dy)
        self.rot = Rotation.from_rotvec(rotvec_x) * Rotation.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        d = np.array([dx, -dy, dz])  # the y axis is flipped
        self.look_at += 2 * self.rot.as_matrix()[:3, :3] @ d * self.radius / self.image_height * math.tan(np.radians(self.fovy) / 2)


import os
import tempfile

import cv2
import numpy as np

# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # osmesa or egl
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
import trimesh
from psbody.mesh import Mesh


class MeshRenderer:
    def __init__(self, size, fov=16 / 180 * np.pi, camera_pose=None, light_pose=None, black_bg=False):
        # Camera
        self.frustum = {'near': 0.01, 'far': 3.0}
        self.camera = pyrender.PerspectiveCamera(yfov=fov, znear=self.frustum['near'],
                                                 zfar=self.frustum['far'], aspectRatio=1.0)

        # Material
        self.primitive_material = pyrender.material.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=[0.3, 0.3, 0.3, 1.0],
            metallicFactor=0.8,
            roughnessFactor=0.8
        )

        # Lighting
        light_color = np.array([1., 1., 1.])
        self.light = pyrender.DirectionalLight(color=light_color, intensity=2)
        self.light_angle = np.pi / 6.0

        # Scene
        self.scene = None
        self._init_scene(black_bg)

        # add camera and lighting
        self._init_camera(camera_pose)
        self._init_lighting(light_pose)

        # Renderer
        self.renderer = pyrender.OffscreenRenderer(*size, point_size=1.0)

    def _init_scene(self, black_bg=False):
        if black_bg:
            self.scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
        else:
            self.scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])

    def _init_camera(self, camera_pose=None):
        if camera_pose is None:
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = np.array([0, 0, 1])
        self.camera_pose = camera_pose.copy()
        self.camera_node = self.scene.add(self.camera, pose=camera_pose)

    def _init_lighting(self, light_pose=None):
        if light_pose is None:
            light_pose = np.eye(4)
            light_pose[:3, 3] = np.array([0, 0, 1])
        self.light_pose = light_pose.copy()

        light_poses = self._get_light_poses(self.light_angle, light_pose)
        self.light_nodes = [self.scene.add(self.light, pose=light_pose) for light_pose in light_poses]

    def set_camera_pose(self, camera_pose):
        self.camera_pose = camera_pose.copy()
        self.scene.set_pose(self.camera_node, pose=camera_pose)

    def set_lighting_pose(self, light_pose):
        self.light_pose = light_pose.copy()

        light_poses = self._get_light_poses(self.light_angle, light_pose)
        for light_node, light_pose in zip(self.light_nodes, light_poses):
            self.scene.set_pose(light_node, pose=light_pose)

    def render_mesh(self, mesh, t_center, rot=np.zeros(3), tex_img=None, tex_uv=None,
                    camera_pose=None, light_pose=None):
        # Prepare mesh
        mesh = Mesh(mesh.v, mesh.f)  # clone mesh
        mesh.v[:] = cv2.Rodrigues(rot)[0].dot((mesh.v - t_center).T).T + t_center
        if tex_img is not None:
            tex = pyrender.Texture(source=tex_img, source_channels='RGB')
            tex_material = pyrender.material.MetallicRoughnessMaterial(baseColorTexture=tex)
            mesh.vt, mesh.ft = tex_uv['vt'], tex_uv['ft']
            tri_mesh = self._pyrender_mesh_workaround(mesh)
            render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=tex_material)
        else:
            tri_mesh = trimesh.Trimesh(vertices=mesh.v, faces=mesh.f)
            render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=self.primitive_material, smooth=True)
        mesh_node = self.scene.add(render_mesh, pose=np.eye(4))

        # Change camera and lighting pose if necessary
        if camera_pose is not None:
            self.set_camera_pose(camera_pose)
        if light_pose is not None:
            self.set_lighting_pose(light_pose)

        # Render
        flags = pyrender.RenderFlags.SKIP_CULL_FACES
        color, depth = self.renderer.render(self.scene, flags=flags)

        # Remove mesh
        self.scene.remove_node(mesh_node)

        return color, depth

    @staticmethod
    def _get_light_poses(light_angle, light_pose):
        light_poses = []
        init_pos = light_pose[:3, 3].copy()

        light_poses.append(light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([light_angle, 0, 0]))[0].dot(init_pos)
        light_poses.append(light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([-light_angle, 0, 0]))[0].dot(init_pos)
        light_poses.append(light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -light_angle, 0]))[0].dot(init_pos)
        light_poses.append(light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([0, light_angle, 0]))[0].dot(init_pos)
        light_poses.append(light_pose.copy())

        return light_poses

    @staticmethod
    def _pyrender_mesh_workaround(mesh):
        # Workaround as pyrender requires number of vertices and uv coordinates to be the same
        with tempfile.NamedTemporaryFile(suffix='.obj') as f:
            mesh.write_obj(f.name)
            tri_mesh = trimesh.load(f.name, process=False)
        return tri_mesh
