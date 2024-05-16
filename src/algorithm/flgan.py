from .fedavg import FedavgOptimizer



class FlganOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(FlganOptimizer, self).__init__(params=params, **kwargs)
