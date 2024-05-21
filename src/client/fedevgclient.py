import copy
import torch
import logging

from .fedavgclient import FedavgClient
from src import MetricManager

logger = logging.getLogger(__name__)



class FedevgClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedevgClient, self).__init__(**kwargs)
        self.inputs_synth = None
        self.targets_synth = None

        self.sigma = 0.01
        self.alpha = 1.
        self.ld_steps = 20

    @torch.enable_grad()
    def energy_gradient(self, x, y):
        self.model.eval()

        x = x.detach().clone()
        x.requires_grad_(True)
        
        # calculate the gradient
        x_energy = -self.model(x).gather(1, y.view(-1, 1))
        x_grad = torch.autograd.grad(
            outputs=x_energy.sum(),
            inputs=[x],
            only_inputs=True
        )[0]
        self.model.train()
        return x_grad.detach(), x_energy.detach()

    def langevine_dynamics_step(self, x_old, y, thres=3.):
        energy_grad, _ = self.energy_gradient(x_old, y)
        if thres is not None:
            energy_grad = torch.clip(energy_grad, -thres, thres)
        epsilon = torch.randn_like(energy_grad)
        x_new = x_old - self.alpha * energy_grad + epsilon * self.sigma
        return x_new

    def sample(self, num_samples, device):
        # radnomly sample from buffer
        y = torch.randint(0, self.args.num_classes, (num_samples,)).to(device)
        idx = torch.randint(0, len(self.inputs_synth) // self.args.num_classes, (num_samples,))
        aug_idx = y.cpu() * (len(self.inputs_synth) // self.args.num_classes) + idx

        # initialize persistent samples
        x_ebm = self.inputs_synth[aug_idx].to(device)
        y_ebm = self.targets_synth[aug_idx].to(device)

        # MALA update
        grad, energy_old = self.energy_gradient(x_ebm, y_ebm)
        for _ in range(self.ld_steps):
            x_proposal = x_ebm - self.alpha * grad + torch.randn_like(grad).mul(self.sigma)

            grad_new, energy_new = self.energy_gradient(x_ebm, y_ebm)

            log_xhat_given_x = -1.0 * ((x_proposal - x_ebm - self.alpha * grad) ** 2).sum() / (2 * self.alpha**2)
            log_x_given_xhat = -1.0 * ((x_ebm - x_proposal - self.alpha * grad_new) ** 2).sum() / (2 * self.alpha**2)
            log_alpha = energy_new - energy_old + log_x_given_xhat - log_xhat_given_x 
            
            # acceptance
            accept_indices = torch.where(torch.log(torch.rand_like(log_alpha)) < log_alpha.detach(), 1, 0).cumsum(0).squeeze().sub(1).unique()
            x_ebm[accept_indices] = x_proposal[accept_indices]
            energy_old[accept_indices] = energy_new[accept_indices]
            grad[accept_indices] = grad_new[accept_indices]
        return x_ebm.clip(0., 1.), y_ebm

    @torch.enable_grad()
    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)

        optimizer = self.optim(
            list(param for param in self.model.parameters() if param.requires_grad), 
            **self._refine_optim_args(self.args)
        )

        for e in range(self.args.E):
            for inputs_ce, targets_ce in self.train_loader:
                inputs_ce, targets_ce = inputs_ce.to(self.args.device), targets_ce.to(self.args.device)
                inputs_pcd, targets_pcd = self.sample(inputs_ce.size(0), self.args.device)

                outputs_ce = self.model(inputs_ce)
                outputs_pcd = self.model(inputs_pcd.detach())
                
                e_pos = outputs_ce.gather(1, targets_ce.view(-1, 1))
                e_neg = self.model(inputs_pcd.detach()).gather(1, targets_pcd.view(-1, 1))
                pcd_loss = -(e_pos - e_neg).mean()
                ce_loss = self.criterion(outputs_ce, targets_ce) + self.criterion(outputs_pcd, targets_pcd).mul(0.1)
                loss = pcd_loss + ce_loss 

                optimizer.zero_grad()
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(param for param in self.model.parameters() if param.requires_grad), 
                        self.args.max_grad_norm
                    )
                optimizer.step()

                # collect clf results
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
        return mm.results

    @torch.no_grad()
    def evaluate(self):
        if self.args.train_only: # `args.test_size` == 0
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs_ce, targets_ce in self.test_loader:
            inputs_ce, targets_ce = inputs_ce.to(self.args.device), targets_ce.to(self.args.device)
            inputs_pcd, targets_pcd = self.sample(inputs_ce.size(0), self.args.device)

            outputs_ce = self.model(inputs_ce)
            outputs_pcd = self.model(inputs_pcd.detach())
            
            e_pos = outputs_ce.gather(1, targets_ce.view(-1, 1))
            e_neg = self.model(inputs_pcd.detach()).gather(1, targets_pcd.view(-1, 1))
            pcd_loss = -(e_pos - e_neg).mean()
            ce_loss = self.criterion(outputs_ce, targets_ce) + self.criterion(outputs_pcd, targets_pcd).mul(0.1)
            loss = pcd_loss + ce_loss 

            # collect clf results
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
            self.model.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results

    @torch.no_grad()
    def evaluate_classifier(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs_ce, targets_ce in self.test_loader:
            inputs_ce, targets_ce = inputs_ce.to(self.args.device), targets_ce.to(self.args.device)
            
            outputs = self.model(inputs_ce)
            loss = self.criterion(outputs, targets_ce)

            mm.track(loss.item(), outputs.detach().cpu(), targets_ce.detach().cpu())
        else:
            self.model.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results
    
    def upload(self):
        energy_grad, energy = self.energy_gradient(self.inputs_synth.cpu(), self.targets_synth.cpu())
        return energy_grad, energy.sign().mul(-1).exp()