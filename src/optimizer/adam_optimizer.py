from base import BaseOptimizer
import torch.optim as optim

class AdamOptimizer(BaseOptimizer):
    def get_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)