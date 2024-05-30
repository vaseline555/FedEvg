from .basealgorithm import BaseOptimizer



class FedcvaekdOptimizer(BaseOptimizer):
    def __init__(self, params, **kwargs):
        super(FedcvaekdOptimizer, self).__init__(params=params, **kwargs)

    def step(self):
        pass

    def accumulate(self):
        pass
