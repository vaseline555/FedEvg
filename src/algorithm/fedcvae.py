from .basealgorithm import BaseOptimizer



class FedcvaeOptimizer(BaseOptimizer):
    def __init__(self, params, **kwargs):
        super(FedcvaeOptimizer, self).__init__(params=params, **kwargs)

    def step(self):
        pass

    def accumulate(self):
        pass
