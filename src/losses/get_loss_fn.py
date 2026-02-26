from .motion_loss import MotionLoss
from .simple_loss import SimpleLoss


def get_loss_fn(loss_type):
    if loss_type.lower() == 'motion':
        return MotionLoss()
    elif loss_type.lower() == 'simple':
        return SimpleLoss()
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")