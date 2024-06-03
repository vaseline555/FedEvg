import copy
import torch
import logging

from itertools import islice

from .fedavgclient import FedavgClient
from src import MetricManager, init_weights

logger = logging.getLogger(__name__)



class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

class FedevgClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedevgClient, self).__init__(**kwargs)
        self.synth_dataset = None
        self.classifier = None
        self.have_ckpt = False

    @torch.enable_grad()
    def energy_gradient(self, x, y):
        self.model.eval()

        # calculate the gradient
        x = x.detach().clone()
        x.requires_grad_(True)
        x_energy = -self.model(x).gather(1, y.view(-1, 1))
        x_grad = torch.autograd.grad(
            outputs=x_energy.sum(),
            inputs=[x],
            only_inputs=True
        )[0]
        self.model.train()
        return x_grad.detach(), x_energy.detach()

    def langevine_dynamics_step(self, x_old, y):
        energy_grad, _ = self.energy_gradient(x_old, y)
        if self.args.ld_threshold is not None:
            energy_grad = torch.clip(energy_grad, -self.args.ld_threshold, self.args.ld_threshold)
        epsilon = torch.randn_like(energy_grad)
        x_new = x_old - self.args.alpha * energy_grad + epsilon * self.args.sigma
        return x_new

    def sample(self, inputs, targets):
        # initial proposal
        if self.args.cd_init == 'noise':
            x_ebm, y_ebm = torch.randn_like(inputs), targets.clone()
        elif self.args.cd_init == 'cd':
            x_ebm, y_ebm = next(iter(self.train_loader))
            x_ebm, y_ebm = x_ebm.to(self.args.device), y_ebm.to(self.args.device)
        elif self.args.cd_init == 'pcd':
            x_ebm, y_ebm = inputs.clone(), targets.clone()

        # MCMC sampling
        if self.args.mcmc == 'ula':
            for _ in range(self.args.ld_steps):
                x_ebm = x_ebm - self.args.alpha * self.energy_gradient(x_ebm, y_ebm)[0] \
                    + torch.randn_like(x_ebm).mul(self.args.sigma)
        elif self.args.mcmc == 'mala':
            grad, energy_old = self.energy_gradient(x_ebm, y_ebm)
            for _ in range(self.args.ld_steps):
                x_proposal = x_ebm - self.args.alpha * grad \
                    + torch.randn_like(grad).mul(self.args.sigma)

                grad_new, energy_new = self.energy_gradient(x_ebm, y_ebm)

                log_xhat_given_x = -1.0 * ((x_proposal - x_ebm - self.args.alpha * grad)**2).sum() \
                    / (2 * self.args.alpha**2)
                log_x_given_xhat = -1.0 * ((x_ebm - x_proposal - self.args.alpha * grad_new)**2).sum() \
                    / (2 * self.args.alpha**2)
                log_alpha = energy_new - energy_old + log_x_given_xhat - log_xhat_given_x 
                
                # acceptance
                accept_indices = torch.where(
                    torch.log(torch.rand_like(log_alpha)) < log_alpha.detach(), 
                    1, 
                    0
                ).cumsum(0).squeeze().sub(1).unique()
                x_ebm[accept_indices] = x_proposal[accept_indices]
                energy_old[accept_indices] = energy_new[accept_indices]
                grad[accept_indices] = grad_new[accept_indices]
        return x_ebm.clip(0., 1.).detach(), y_ebm

    @torch.enable_grad()
    def update(self, reinit=False):
        if self.have_ckpt and (self.model is None):
            self.model = torch.load(f"{self.args.ckpt_path}/{self.id}.pt")
        if reinit:
            init_weights(self.model, self.args.init_type, self.args.init_gain)
        if self.args.penult_spectral_norm:
            self.model.classifier.apply(torch.nn.utils.parametrizations.spectral_norm)

        self.model.train()
        self.model.to(self.args.device)
        
        optimizer = self.optim(
            list(param for param in self.model.parameters() if param.requires_grad), 
            **self._refine_optim_args(self.args)
        )

        mm = MetricManager(self.args.eval_metrics)
        for e in range(self.args.E):
            synth_dataset = InfiniteDataLoader(self.synth_dataset, shuffle=True, drop_last=True)
            for inputs, targets in self.train_loader:
                inputs_ce, targets_ce = inputs.to(self.args.device), targets.to(self.args.device)
                inputs_synth, targets_synth = next(islice(synth_dataset, self.args.B))
                inputs_synth, targets_synth = inputs_synth.to(self.args.device), targets_synth.to(self.args.device)
                inputs_pcd, targets_pcd = self.sample(inputs_synth, targets_synth)

                outputs_ce = self.model(inputs_ce)
                outputs_synth = self.model(inputs_synth)

                ce_loss = self.criterion(outputs_ce, targets_ce) + self.criterion(outputs_synth, targets_synth).mul(self.args.ce_lambda)
                outputs_pcd = self.model(inputs_pcd)
                e_pos = -outputs_ce.gather(1, targets_ce.view(-1, 1)).mean()
                e_neg = -outputs_pcd.gather(1, targets_pcd.view(-1, 1)).mean()
                pcd_loss = (e_pos - e_neg)

                loss = pcd_loss + ce_loss

                optimizer.zero_grad()
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(param for param in self.model.parameters() if param.requires_grad), 
                        self.args.max_grad_norm
                    )
                optimizer.step()

                # collect results
                mm.track(
                    loss=[
                        ce_loss.detach().cpu().item(), 
                        pcd_loss.detach().cpu().item()
                    ], 
                    pred=inputs_pcd.detach().cpu().clip(0., 1.),
                    true=inputs_ce.detach().cpu(), 
                    suffix=['ce', 'pcd'],
                    calc_fid=False
                )
                mm.track(loss.item(), outputs_ce.detach().cpu(), targets_ce.detach().cpu())
            else:
                mm.aggregate(len(self.training_set), e + 1)
        else:
            self.model.to('cpu')
            if self.args.penult_spectral_norm:
                torch.nn.utils.parametrize.remove_parametrizations(self.model.classifier, 'weight')
            torch.save(self.model, f"{self.args.ckpt_path}/{self.id}.pt")
            self.have_ckpt = True
        return mm.results

    @torch.no_grad()
    def evaluate(self):
        if self.args.train_only: # `args.test_size` == 0
            return {'loss': -1, 'metrics': {'none': -1}}
        if self.have_ckpt and (self.model is None):
            self.model = torch.load(f"{self.args.ckpt_path}/{self.id}.pt")
        if self.args.penult_spectral_norm:
            self.model.classifier.apply(torch.nn.utils.parametrizations.spectral_norm)
        self.model.eval()
        self.model.to(self.args.device)

        mm = MetricManager(self.args.eval_metrics)
        synth_dataset = InfiniteDataLoader(self.synth_dataset, shuffle=True, drop_last=True)
        for inputs_ce, targets_ce in self.test_loader:
            inputs_ce, targets_ce = inputs_ce.to(self.args.device), targets_ce.to(self.args.device)
            inputs_synth, targets_synth = next(islice(synth_dataset, self.args.B))
            inputs_synth, targets_synth = inputs_synth.to(self.args.device), targets_synth.to(self.args.device)
            inputs_pcd, targets_pcd = self.sample(inputs_synth, targets_synth)

            outputs_ce = self.model(inputs_ce)
            outputs_pcd = self.model(inputs_pcd.detach())
            
            e_pos = -outputs_ce.gather(1, targets_ce.view(-1, 1)).mean()
            e_neg = -outputs_pcd.gather(1, targets_pcd.view(-1, 1)).mean()
            pcd_loss = (e_pos - e_neg)
            ce_loss = self.criterion(outputs_ce, targets_ce)

            # collect results
            mm.track(
                loss=[
                    ce_loss.detach().cpu().item(), 
                    pcd_loss.detach().cpu().item()
                ], 
                pred=inputs_pcd.detach().cpu().clip(0., 1.),
                true=inputs_ce.detach().cpu(), 
                suffix=['ce', 'pcd'],
                calc_fid=False
            )
            mm.track(ce_loss.item(), outputs_ce.detach().cpu(), targets_ce.detach().cpu())
        else:
            self.model.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results

    @torch.no_grad()
    def evaluate_classifier(self):
        self.classifier.eval()
        self.classifier.to(self.args.device)

        mm = MetricManager(self.args.eval_metrics)
        for inputs_ce, targets_ce in self.test_loader:
            inputs_ce, targets_ce = inputs_ce.to(self.args.device), targets_ce.to(self.args.device)
            
            outputs = self.classifier(inputs_ce)
            loss = self.criterion(outputs, targets_ce)

            mm.track(loss.item(), outputs.detach().cpu(), targets_ce.detach().cpu())
        else:
            self.classifier.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results
    
    def upload(self):
        inputs_synth, targets_synth = self.synth_dataset.tensors
        energy_grad, energy = self.energy_gradient(inputs_synth.cpu(), targets_synth.cpu())
        return energy_grad, energy.sign().mul(-1).exp()
        