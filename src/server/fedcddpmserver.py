import torch
import logging
import numpy as np

from PIL import Image
from collections import defaultdict

from src import MetricManager
from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

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

class FedcddpmServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FedcddpmServer, self).__init__(**kwargs)
        classifier = torch.nn.Sequential(
            torch.nn.Conv2d(self.args.in_channels, self.args.hidden_size, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(self.args.hidden_size, self.args.hidden_size * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.args.hidden_size * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(self.args.hidden_size * 2, self.args.hidden_size * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.args.hidden_size * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(self.args.hidden_size * 4 * (self.args.resize // 8)**2, self.args.num_classes)
        )
        self.classifier = self._init_model(classifier)
        self.results['communication_bits'] = sum(p.numel() for p in self.global_model.parameters())

    def _log_results(self, resulting_sizes, results, eval, participated, save_raw):
        losses, losses_D, losses_G, metrics, num_samples = list(), list(), list(), defaultdict(list), list()
        generated = list()

        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [CLIENT] < {str(identifier).zfill(6)} > '
            if eval: # get loss and metrics
                raise NotImplementedError(f'{self.args.algorithm} does not support local evaluation... please check!') 
            else: # same, but retireve results of last epoch's
                # loss
                loss = result[self.args.E]['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)

                # metrics
                for name, value in result[self.args.E]['metrics'].items():
                    client_log_string += f'| {name}: {value:.4f} '
                    metrics[name].append(value)                
            # get sample size
            num_samples.append(resulting_sizes[identifier])

            # log per client
            logger.info(client_log_string)
        else:
            num_samples = np.array(num_samples).astype(float)

        # aggregate into total logs
        result_dict = defaultdict(dict)
        total_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [SUMMARY] ({len(resulting_sizes)} clients):'

        # loss
        losses_array = np.array(losses).astype(float)
        weighted = losses_array.dot(num_samples) / sum(num_samples); std = losses_array.std()

        total_log_string += f'\n    - Loss: Avg. ({weighted:.4f}) Std. ({std:.4f})'
        result_dict['loss'] = {'avg': weighted.astype(float), 'std': std.astype(float)}

        if save_raw:
            result_dict['loss']['raw'] = losses

        self.writer.add_scalars(
            f'Local {"Test" if eval else "Training"} Loss' + eval * f' ({"In" if participated else "Out"})',
            {'Avg.': weighted, 'Std.': std},
            self.round
        )

        # metrics
        for name, val in metrics.items():
            val_array = np.array(val).astype(float)
            weighted = val_array.dot(num_samples) / sum(num_samples); std = val_array.std()

            total_log_string += f'\n    - {name.title()}: Avg. ({weighted:.4f}) Std. ({std:.4f})'
            result_dict[name] = {'avg': weighted.astype(float), 'std': std.astype(float)}
                
            if save_raw:
                result_dict[name]['raw'] = val

            self.writer.add_scalars(
                f'Local {"Test" if eval else "Training"} {name.title()}' + eval * f' ({"In" if participated else "Out"})',
                {'Avg.': weighted, 'Std.': std},
                self.round
            )

        self.writer.flush()
        logger.info(total_log_string)
        return result_dict

    @torch.no_grad()
    def _central_evaluate(self):
        mm = MetricManager(self.args.eval_metrics)
        
        sampler = GaussianDiffusionSampler(self.global_model)
        sampler.to(self.args.device)

        inputs_synth, targets_synth = [], []
        for inputs, targets in torch.utils.data.DataLoader(
            dataset=self.server_dataset, 
            batch_size=self.args.B, 
            shuffle=False
        ):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            noises = torch.randn_like(inputs)
            generated = sampler(noises, targets)

            # collect
            inputs_synth.append(generated.detach().cpu())
            targets_synth.append(targets.detach().cpu())
        else:
            sampler.to('cpu')

        # generated images
        gen_imgs = generated.mul(0.5).add(0.5).detach().cpu().numpy()
        viz_idx = np.random.randint(0, len(gen_imgs), size=(), dtype=int)
        to_viz = (gen_imgs[viz_idx] * 255).astype(np.uint8)
        to_viz = np.transpose(to_viz, (1, 2, 0))
        self.writer.add_image(
            'Server Generated Image', 
            to_viz, 
            self.round // self.args.eval_every,
            dataformats='HWC'
        )

        # save server-side synthetic dataset
        inputs_synth = torch.cat(inputs_synth).mul(0.5).add(0.5).mul(255).numpy().astype(np.uint8)
        targets_synth = torch.cat(targets_synth).numpy().astype(int)
        np.savez(
            f'{self.args.result_path}/server_generated_{str(self.round).zfill(4)}.npz', 
            inputs=inputs_synth,
            targets=targets_synth
        )
