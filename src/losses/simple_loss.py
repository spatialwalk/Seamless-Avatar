import torch.nn as nn


class SimpleLoss():
    def __init__(self):
        self.loss_fn = nn.MSELoss()
      

    def __call__(self, output):
        pred = output['pred']
        gt = output['gt']
        
        loss_dict = {}
        loss_dict['total_loss'] = self.loss_fn(pred, gt)
        
        return loss_dict
