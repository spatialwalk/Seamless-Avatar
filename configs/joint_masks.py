import numpy as np
joint_mask_lower = [
    True, True, True, False, True, True, False, True, True, False,
    True, True, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False
][1:22]
joint_mask_upper = ~np.bool(joint_mask_lower)

JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
][1:22]

joint_mask_arms = [False]*len(JOINT_NAMES)
for i in range(len(JOINT_NAMES)):
    if JOINT_NAMES[i] in ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
        joint_mask_arms[i] = True
        
joint_mask_arms = joint_mask_upper
