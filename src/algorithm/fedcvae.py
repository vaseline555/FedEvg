from .fedavg import FedavgOptimizer



class FedcvaeOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(FedcvaeOptimizer, self).__init__(params=params, **kwargs)
