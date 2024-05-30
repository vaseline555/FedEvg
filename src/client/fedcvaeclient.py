import torch
import logging

from .fedavgclient import FedavgClient
from src import MetricManager

logger = logging.getLogger(__name__)



class FedcvaeClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedcvaeClient, self).__init__(**kwargs)
        self.classifier = None

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
            for inputs, targets in self.train_loader:
                # real image and label
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                # one-hot encode label
                targets_hot = torch.nn.functional.one_hot(targets, self.args.num_classes) 
                
                # inference
                generated, mu, std = self.model(inputs, targets_hot)

                # Calculate losses
                recon_loss = self.criterion(generated, inputs)
                kld_loss = -0.5 * (1 + std.pow(2).log() - mu.pow(2) - std.pow(2)).sum(1).mean()
                loss = recon_loss + kld_loss

                # backward
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
                        recon_loss.detach().cpu().item(), 
                        kld_loss.detach().cpu().item()
                    ], 
                    pred=generated.detach().cpu(),
                    true=inputs.detach().cpu(), 
                    suffix=['recon', 'kl'],
                    calc_fid=False
                )
                mm.track(0, torch.randn(len(targets), self.args.num_classes).detach().cpu(), targets.detach().cpu()) # dummy
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

        for inputs, targets in self.test_loader:
            # real image and label
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            # one-hot encode label
            targets_hot = torch.nn.functional.one_hot(targets, self.args.num_classes)
            
            # inference
            generated, mu, std = self.model(inputs, targets_hot)

            # Calculate losses
            recon_loss = self.criterion(generated, inputs)
            kld_loss = -0.5 * (1 + std.pow(2).log() - mu.pow(2) - std.pow(2)).sum(1).mean()
            loss = recon_loss + kld_loss

            # collect clf results
            mm.track(
                loss=[
                    recon_loss.detach().cpu().item(), 
                    kld_loss.detach().cpu().item()
                ], 
                pred=generated.detach().cpu(),
                true=inputs.detach().cpu(), 
                suffix=['recon', 'kl'],
                calc_fid=False
            )
            mm.track(loss.item(), torch.randn(len(targets), self.args.num_classes).detach().cpu(), targets.detach().cpu()) # dummy
        else:
            self.model.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results

    @torch.no_grad()
    def evaluate_classifier(self):
        mm = MetricManager(self.args.eval_metrics)
        self.classifier.eval()
        self.classifier.to(self.args.device)

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            
            outputs = self.classifier(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)

            mm.track(loss.item(), outputs.detach().cpu(), targets.detach().cpu())
        else:
            mm.aggregate(len(self.test_set))
            self.classifier.to('cpu')
        return mm.results