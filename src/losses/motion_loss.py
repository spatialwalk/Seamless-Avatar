import torch.nn as nn


class MotionLoss():
    def __init__(self):
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()
        # self.loss_fn = nn.functional.l1_loss

    def __call__(self, motion_pred, motion_gt):

        motion_pred_vel = motion_pred[:, 1:] - motion_pred[:, :-1]
        motion_gt_vel = motion_gt[:, 1:] - motion_gt[:, :-1]

        loss_v = self.loss_fn(motion_pred_vel, motion_gt_vel)
        loss_x = self.loss_fn(motion_pred, motion_gt)

        return loss_x, loss_v
