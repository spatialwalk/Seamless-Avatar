import torch
import torch.nn.functional as F
import utils.tgm_conversion as tgm


def matrot2aa(pose_matrot):
    '''
    :param pose_matrot: Nx3x3
    :return: Nx3
    '''
    bs = pose_matrot.size(0)
    homogen_matrot = F.pad(pose_matrot, [0, 1])
    pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot)
    return pose


def aa2matrot(pose):
    '''
    :param Nx3
    :return: pose_matrot: Nx3x3
    '''
    bs = pose.size(0)
    num_joints = pose.size(1) // 3
    pose_body_matrot = tgm.angle_axis_to_rotation_matrix(
        pose)[:, :3, :3].contiguous()  # .view(bs, num_joints*9)
    return pose_body_matrot


def matrot2sixd(pose_matrot):
    """
    :param pose_matrot: Nx3x3
    :return: pose_6d: Nx6
    """
    pose_6d = torch.cat([pose_matrot[:, :3, 0], pose_matrot[:, :3, 1]], dim=1)
    return pose_6d


def matrot2sixd_single(pose_matrot):
    """
    :param pose_matrot: 3x3
    :return: pose_6d: 6
    """
    pose_6d = torch.cat([pose_matrot[:, 0], pose_matrot[:, 1]], dim=-1)
    return pose_6d


def aa2sixd(pose_aa,batch=False):
    """
    :param pose_aa Nx3
    :return: pose_6d: Nx6
    """
    if batch:
        B, J, C = pose_aa.shape
        assert C == 3, f"pose_aa.shape: {pose_aa.shape} != (B, J, 3)"
        pose_aa = pose_aa.reshape(-1, 3)
    pose_matrot = aa2matrot(pose_aa)
    pose_6d = matrot2sixd(pose_matrot)
    if batch:
        pose_6d = pose_6d.reshape(B, J, 6)
    return pose_6d


def sixd2matrot(pose_6d):
    """
    :param pose_6d: Nx6
    :return: pose_matrot: Nx3x3
    """
    bs = pose_6d.shape
    rot_vec_1 = pose_6d[:, :3]
    rot_vec_1 = rot_vec_1 / rot_vec_1.norm(dim=-1).reshape(bs[0], 1)
    rot_vec_2 = pose_6d[:, 3:6] - \
        torch.sum(rot_vec_1 * pose_6d[:, 3:6],
                  dim=-1, keepdim=True) * rot_vec_1
    rot_vec_2 = rot_vec_2 / rot_vec_2.norm(dim=-1).reshape(bs[0], 1)
    # rot_vec_2 = pose_6d[:, 3:]
    rot_vec_3 = torch.cross(rot_vec_1, rot_vec_2)  # 叉乘得到第3个基, 共同构成旋转矩阵
    pose_matrot = torch.stack(
        [rot_vec_1, rot_vec_2, rot_vec_3], dim=-1)  # (N, 3, 3)
    return pose_matrot


def sixd2aa(pose_6d, batch=False):
    """
    :param pose_6d: Nx6  (N, 132) -> (N*22, 6) colomn
    :return: pose_aa: Nx3
    """
    if batch:
        B, J, C = pose_6d.shape
        assert C == 6, f"pose_6d.shape: {pose_6d.shape} != (B, J, 6)"
        pose_6d = pose_6d.reshape(-1, 6)
    pose_matrot = sixd2matrot(pose_6d)  # (N, 3, 3)
    pose_aa = matrot2aa(pose_matrot)  # (N, 3)
    if batch:
        pose_aa = pose_aa.reshape(B, J, 3)
    return pose_aa
