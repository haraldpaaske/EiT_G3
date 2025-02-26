from base import BaseOptimizer
import torch.optim as optim

class SGDOptimizer(BaseOptimizer):
    def get_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
