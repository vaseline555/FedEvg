import torch
import logging

import numpy as np

from .fedavgclient import FedavgClient
from src import MetricManager

logger = logging.getLogger(__name__)



def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer(torch.nn.Module):
    def __init__(self, model, beta_1=0.0001, beta_T=0.028, T=500):
        super().__init__()
        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, y):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        pred = self.model(x_t, t, y)
        return pred, noise

class FedcddpmClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedcddpmClient, self).__init__(**kwargs)
        
    @torch.enable_grad()
    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        trainer = GaussianDiffusionTrainer(self.model)
        trainer.train()
        trainer.to(self.args.device)

        optimizer = self.optim(
            list(param for param in self.model.parameters() if param.requires_grad), 
            **self._refine_optim_args(self.args)
        )

        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                # real image and label
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                if torch.rand(1) < 0.1:
                    targets = torch.zeros_like(targets)
                preds, noises = trainer(inputs, targets)
                loss = self.criterion(preds, noises)

                optimizer.zero_grad()
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(param for param in self.model.generator.parameters() if param.requires_grad), 
                        self.args.max_grad_norm
                    )
                optimizer.step()

                # collect clf results
                mm.track(loss.detach().cpu(), preds.mean(0), noises.mean(0))
            else:
                mm.aggregate(len(self.training_set), e + 1)
        else:
            trainer.to('cpu')
        return mm.results

    @torch.no_grad()
    def evaluate(self):
        return {'loss': -1, 'metrics': {'none': -1}}
