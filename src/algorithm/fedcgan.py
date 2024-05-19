from .fedavg import FedavgOptimizer



class FedcganOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(FedcganOptimizer, self).__init__(params=params, **kwargs)
