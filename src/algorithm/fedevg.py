from .fedavg import FedavgOptimizer



class FeddmOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(FeddmOptimizer, self).__init__(params=params, **kwargs)
