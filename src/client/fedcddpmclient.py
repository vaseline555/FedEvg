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
        
class GaussianDiffusionSampler(torch.nn.Module):
    def __init__(self, model, beta_1=0.0001, beta_T=0.028, T=100, w=1.):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.nn.functional.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape, f"x_t: {x_t.shape} != {eps.shape} :eps"
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, y):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        cond_eps = self.model(x_t, t, y)
        uncond_eps = self.model(x_t, t, torch.zeros_like(y).to(y.device))
        eps = (1. + self.w) * cond_eps - self.w * uncond_eps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, y):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t, y=y)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   

class FedcddpmClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedcddpmClient, self).__init__(**kwargs)
        
    @torch.enable_grad()
    def update(self):
        trainer = GaussianDiffusionTrainer(self.model)
        trainer.train()
        trainer.to(self.args.device)

        optimizer = self.optim(
            list(param for param in self.model.parameters() if param.requires_grad), 
            **self._refine_optim_args(self.args)
        )

        mm = MetricManager(self.args.eval_metrics)
        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
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
                mm.track(
                    loss=[loss.detach().cpu().item(), -1], 
                    pred=preds.detach().cpu(),
                    true=inputs.detach().cpu(), 
                    suffix=['recon', 'none'],
                    calc_fid=False
                )
                mm.track(loss.item(), torch.randn(len(targets), self.args.num_classes).detach().cpu(), targets.detach().cpu()) # dummy
            else:
                mm.aggregate(len(self.training_set), e + 1)
        else:
            trainer.to('cpu')
        return mm.results

    @torch.no_grad()
    def evaluate(self):
        if self.args.train_only: # `args.test_size` == 0
            return {'loss': -1, 'metrics': {'none': -1}}

        sampler = GaussianDiffusionSampler(self.model, T=10)
        sampler.to(self.args.device)

        mm = MetricManager(self.args.eval_metrics)
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            noises = torch.randn_like(inputs)
            generated = sampler(noises, targets)

            loss = self.criterion(generated, noises)

            # collect clf results
            mm.track(
                loss=[loss.detach().cpu().item(), -1], 
                pred=generated.detach().cpu(),
                true=inputs.detach().cpu(), 
                suffix=['recon', 'none'],
                calc_fid=False
            )
            mm.track(loss.item(), torch.randn(len(targets), self.args.num_classes).detach().cpu(), targets.detach().cpu()) # dummy
        else:
            mm.aggregate(len(self.test_set))
            sampler.to('cpu')
        return mm.results