import copy
import torch
import logging

from .fedavgclient import FedavgClient
from src import MetricManager

logger = logging.getLogger(__name__)



class FeddmClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FeddmClient, self).__init__(**kwargs)
        self.inputs_synth = None
        self.targets_synth = None

    def _random_perturb(self, net):
        for p in net.parameters():
            gauss = torch.normal(mean=torch.zeros_like(p), std=1)
            if p.grad is None:
                p.grad = gauss
            else:
                p.grad.data.copy_(gauss.data)
        
        norm = torch.norm(
            torch.stack([
                (p.grad).norm(p=2) for p in net.parameters() if p.grad is not None
            ]), p=2
        )
        with torch.no_grad():
            scale = 5.0 / (norm + 1e-12)
            scale = torch.clamp(scale, max=1.0)
            for p in net.parameters():
                if p.grad is None: continue
                e_w = 1.0 * p.grad * scale.to(p)
                p.add_(e_w)
        net.zero_grad()
        return net

    @torch.enable_grad()
    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        model = copy.deepcopy(self.model)
        model.train()
        model.to(self.args.device)
        
        samples_per_class = 10000 // int(self.args.C * self.args.K) // self.args.num_classes
        inputs_synth = torch.randn(self.args.num_classes * samples_per_class, self.args.in_channels, self.args.resize, self.args.resize).to(self.args.device)
        targets_synth = torch.cat([torch.ones(samples_per_class).mul(c) for c in range(self.args.num_classes)]).view(-1).long().to(self.args.device)

        optimizer = self.optim([inputs_synth], **self._refine_optim_args(self.args))
        for e in range(self.args.E):
            model = self._random_perturb(copy.deepcopy(model))
            for param in model.parameters():
                param.requires_grad = False

            for inputs, targets in self.train_loader:
                # real image and label
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                
                synths, indices = [], []
                for c in targets:
                    idx = torch.randint(c * samples_per_class, (c + 1) * samples_per_class, ())
                    synths.append(inputs_synth[idx])
                    indices.append(idx.item())
                synths = torch.stack(synths)
                synths.requires_grad_(True)

                real_features = model.features(inputs).detach()
                synth_features = model.features(synths)

                real_outputs = model(inputs).detach()
                synth_outputs = model(synths)

                feature_loss = torch.nn.MSELoss()(synth_features, real_features)
                logit_loss = torch.nn.MSELoss()(synth_outputs, real_outputs)
                loss = feature_loss + logit_loss

                optimizer.zero_grad()
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(param for param in self.model.parameters() if param.requires_grad), 
                        self.args.max_grad_norm
                    )
                optimizer.step()
                inputs_synth[indices] = synths.clip(0., 1.).detach().clone()

                # collect clf results
                mm.track(
                    loss=[
                        feature_loss.detach().cpu().item(), 
                        logit_loss.detach().cpu().item()
                    ], 
                    pred=inputs_synth.detach().cpu().clip(0., 1.),
                    true=inputs.detach().cpu(), 
                    suffix=['feature', 'logit'],
                    calc_fid=False
                )
                mm.track(loss.item(), synth_outputs.detach().cpu(), targets.detach().cpu())
            else:
                mm.aggregate(len(self.training_set), e + 1)
        else:
            self.model.to('cpu')
            self.inputs_synth = inputs_synth.detach().clone()
            self.targets_synth = targets_synth.clone()
        return mm.results

    @torch.no_grad()
    def evaluate(self):
        if self.args.train_only: # `args.test_size` == 0
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            mm.track(loss.item(), outputs.detach().cpu(), targets.detach().cpu())
        else:
            self.model.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results