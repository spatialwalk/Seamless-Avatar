import torch
from collections import defaultdict
import tqdm


@torch.no_grad()
def eval_model(model, val_loader, epoch, logger, verbose=False):
    model.eval()
    loss_dict = defaultdict(list)

    if verbose:
        progress_bar = tqdm.tqdm(val_loader, desc=f"val")
    else:
        progress_bar = val_loader

    for batch in progress_bar:

        output = model(batch)
        tmp_loss_dict = output['loss_dict']

        for key, value in tmp_loss_dict.items():
            loss_dict[key].append(value.item())

    loss_dict = dict(sorted(loss_dict.items()))
    for key, value in loss_dict.items():
        value_mean = sum(value) / len(value)
        loss_dict[key] = value_mean

        logger.log({f'val/{key}': float(value_mean)}, step=epoch)

    return loss_dict
