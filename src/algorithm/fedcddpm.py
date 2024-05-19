from .fedavg import FedavgOptimizer



class FedcddpmOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(FedcddpmOptimizer, self).__init__(params=params, **kwargs)
