from .simple_loss import SimpleLoss


def create_loss_fn(config):

    if config.loss_fn_name == 'simple':
        return SimpleLoss()
    else:
        raise ValueError(f"Loss function {config.loss_fn_name} not found")
