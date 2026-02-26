import smplx
import torch
import numpy as np
import pickle
from src.renderer.mesh_render_nvdiffrast import MeshRendererNvdiffrast
from configs import FPS
from configs.joint_masks import joint_mask_lower


def normalize_vertices(vertices):
    assert vertices.shape[2] == 3 and vertices.ndim == 3
    # 计算最大最小坐标
    min_vals = np.min(vertices, axis=(0, 1))
    max_vals = np.max(vertices, axis=(0, 1))
    center = (min_vals + max_vals) / 2.0
    # 移动到中心
    vertices_centered = vertices - center

    # ic(min_vals)
    # 计算最长边长
    scale = np.max(max_vals[:2] - min_vals[:2]) / 2.0
    # 防止除零
    if scale == 0:
        return vertices_centered
    # 归一化，使得最大尺寸在-1~1
    normalized = vertices_centered / scale
    return normalized


class SMPLX_MODEL:
    def __init__(self, model_folder="./models", model_type='smplx'):

        if model_type == 'smplx':
            gender = 'NEUTRAL_2020'
            ext = 'npz'
            self.num_betas = 300
            self.faces = np.load(
                f"{model_folder}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]

        elif model_type == 'smplh':
            gender = 'NEUTRAL'
            ext = 'pkl'
            self.num_betas = 10
            smplh_pkl_path = f"{model_folder}/smplh/SMPLH_NEUTRAL.pkl"
            with open(smplh_pkl_path, 'rb') as body_file:
                self.faces = pickle.load(body_file, encoding='latin1')['f']

        else:
            raise ValueError(f"Invalid model type: {type}")

        num_expression_coeffs = 100
        use_face_contour = False

        # ic(self.faces.min())
        # ic(self.faces.max())

        self.model = smplx.create(model_folder, model_type=model_type, gender=gender, use_face_contour=use_face_contour,
                                  num_betas=self.num_betas, num_expression_coeffs=num_expression_coeffs, ext=ext, use_pca=False)

    @torch.no_grad()
    def get_vertices(
        self,
        beta=None,
        transl=None,
        expression=None,
        jaw_pose=None,
        global_orient=None,
        body_pose=None,
        left_hand_pose=None,
        right_hand_pose=None,
        leye_pose=None,
        reye_pose=None,
    ):
        assert (
            expression is not None or body_pose is not None), "expression or body_pose is required"
        if expression is not None:
            n_frames = expression.shape[0]
        if body_pose is not None:
            n_frames = body_pose.shape[0]

        if global_orient is None:
            global_orient = torch.zeros(
                (n_frames, 3), device=torch.device("cuda"))
        if leye_pose is None:
            leye_pose = torch.zeros((n_frames, 3), device=torch.device("cuda"))
        if reye_pose is None:
            reye_pose = torch.zeros((n_frames, 3), device=torch.device("cuda"))

        if beta is None:
            # beta = torch.ones((n_frame, self.num_betas), device=torch.device("cuda"))*0.5
            beta = torch.zeros((n_frames, self.num_betas),
                               device=torch.device("cuda"))

        output = self.model(betas=beta, transl=transl, expression=expression, jaw_pose=jaw_pose, global_orient=global_orient, body_pose=body_pose,
                            left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, leye_pose=leye_pose, reye_pose=reye_pose, return_verts=True)
        return output["vertices"].cpu().numpy()


def get_data_dict_smplx(smplx_npz_path):

    smplx_data = dict(np.load(smplx_npz_path))

    if 'expression' not in smplx_data:
        exp = np.zeros((smplx_data['body_pose'].shape[0], 100))
    else:
        exp = smplx_data['expression']
        if exp.shape[1] < 100:
            exp = np.concatenate(
                [exp, np.zeros((exp.shape[0], 100 - exp.shape[1]))], axis=1)
        else:
            exp = exp[:, :100]

    if 'jaw_pose' not in smplx_data:
        jawpose = np.zeros((smplx_data['body_pose'].shape[0], 3))
    else:
        jawpose = smplx_data['jaw_pose']
        assert jawpose.shape[1] == 3

    n_frames = exp.shape[0]

    bodypose = torch.from_numpy(smplx_data["body_pose"])[:n_frames]
    bodypose[:, joint_mask_lower] = 0
    bodypose = bodypose.reshape(-1, 63)

    left_hand_pose = torch.from_numpy(
        smplx_data["left_hand_pose"].reshape(-1, 45))[:n_frames]
    right_hand_pose = torch.from_numpy(
        smplx_data["right_hand_pose"].reshape(-1, 45))[:n_frames]

    data_dict = {
        "expression": torch.from_numpy(exp),
        "jaw_pose": torch.from_numpy(jawpose),
        "body_pose": bodypose,
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
    }
    data_dict = {
        k: v.to('cuda').float() for k, v in data_dict.items()
    }
    return data_dict


@torch.no_grad()
def render_smplx_to_video(smplx_npz_path, save_video_path, fps=FPS, smplx_model=None, max_frames=None):

    data_dict = get_data_dict_smplx(smplx_npz_path)

    if smplx_model is None:
        smplx_model = SMPLX_MODEL()
        smplx_model.model.to(torch.device("cuda"))

    vertices = smplx_model.get_vertices(**data_dict)
    if max_frames is not None:
        vertices = vertices[:max_frames]

    vertices = normalize_vertices(vertices)
    # ic(vertices.shape)

    uniform_color = [220, 220, 220, 255]
    light_direction = [0, 0, -1]
    camera_for_body = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.6],
        [0.0, 0.0, 1.0, 10.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    camera_for_face = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 5],
        [0.0, 0.0, 1.0, 10.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    fig_resolution = (720, 480)

    # fig_resolution = (480, 480)
    # vertices *= 6
    # vertices = vertices[:200]
    vertices = vertices * 1.2

    camera = camera_for_body

    mesh_renderer_nvdiffrast = MeshRendererNvdiffrast(fig_resolution)
    # mesh_renderer_nvdiffrast.render_to_img(
    #     vertices[0],
    #     smplx_model.faces,
    #     uniform_color,
    #     camera,
    #     light_direction,
    #     save_img_path='./outputs/smplx_videos/test.png'
    # )
    # exit()
    mesh_renderer_nvdiffrast.render_to_video(
        vertices,
        smplx_model.faces,
        uniform_color,
        camera,
        light_direction,
        save_video_path=save_video_path,
        batch_size=100,
        fps=fps
    )

    return smplx_model
