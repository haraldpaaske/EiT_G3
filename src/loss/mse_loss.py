from torch import nn
from base import BaseLoss

class MSELoss(BaseLoss):
    def get_loss(self):
        return nn.MSELoss()
