from .fedavg import FedavgOptimizer



class FedevgOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(FedevgOptimizer, self).__init__(params=params, **kwargs)

    def step(self):
        pass

    def accumulate(self):
        pass
