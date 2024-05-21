import torch
import logging

from .fedavgclient import FedavgClient
from src import MetricManager

logger = logging.getLogger(__name__)




class FedcganClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedcganClient, self).__init__(**kwargs)
        self.gan_criterion = torch.nn.BCELoss()
        
    @torch.enable_grad()
    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)

        D_optimizer = self.optim(
            list(param for param in self.model.discriminator.parameters() if param.requires_grad), 
            **self._refine_optim_args(self.args)
        )
        G_optimizer = self.optim(
            list(param for param in self.model.generator.parameters() if param.requires_grad), 
            **self._refine_optim_args(self.args)
        )

        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                # real image and label
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                # fake image and label
                fake_label = torch.randint(self.args.num_classes, (inputs.size(0),)).long().to(self.args.device)
                noise = torch.randn(inputs.size(0), self.args.hidden_size * 2, 1, 1).to(self.args.device)
                noise = torch.cat([
                    noise, 
                    torch.eye(self.args.num_classes).to(self.args.device)[
                        targets
                    ].view(-1, self.args.num_classes, 1, 1)
                ], dim=1)

                # update D
                disc_fake, disc_real, clf_fake, clf_real = self.model(noise, inputs, for_D=True)
                
                ## D on real
                D_loss_real = self.gan_criterion(disc_real, torch.ones_like(disc_real))
                clf_loss_real = self.criterion(clf_real, targets)

                ## D on fake
                D_loss_fake = self.gan_criterion(disc_fake, torch.zeros_like(disc_fake))
                clf_loss_fake = self.criterion(clf_fake, torch.ones_like(targets).mul(fake_label).long())

                ## total loss of D
                D_loss = D_loss_real + D_loss_fake + clf_loss_real + clf_loss_fake

                # backward D
                D_optimizer.zero_grad()
                D_loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(param for param in self.model.generator.parameters() if param.requires_grad), 
                        self.args.max_grad_norm
                    )
                D_optimizer.step()

                # update G
                disc_fake, clf_fake, generated = self.model(noise, inputs, for_D=False)

                ## D on fake
                G_loss_fake = self.gan_criterion(disc_fake, torch.ones_like(disc_fake))
                clf_loss_fake = self.criterion(clf_fake, targets)

                ## total loss of D
                G_loss = G_loss_fake + clf_loss_fake

                # backward D
                G_optimizer.zero_grad()
                G_loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(param for param in self.model.generator.parameters() if param.requires_grad), 
                        self.args.max_grad_norm
                    )
                G_optimizer.step()

                # collect clf results
                mm.track(
                    loss=[
                        (D_loss_real + D_loss_fake).detach().cpu().item(), 
                        G_loss_fake.detach().cpu().item()
                    ], 
                    pred=generated.detach().cpu(),
                    true=inputs.detach().cpu(), 
                    suffix=['D', 'G'],
                    calc_fid=False
                )
                mm.track(clf_loss_real.detach().cpu().item(), clf_real.detach().cpu(), targets)
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

            # fake image and label
            fake_label = torch.randint(self.args.num_classes, (inputs.size(0),)).long().to(self.args.device)
            noise = torch.randn(inputs.size(0), self.args.hidden_size * 2, 1, 1).to(self.args.device)
            noise = torch.cat([
                noise, 
                torch.eye(self.args.num_classes).to(self.args.device)[
                    targets
                ].view(-1, self.args.num_classes, 1, 1)
            ], dim=1)

            # D
            disc_fake, disc_real, clf_fake, clf_real = self.model(noise, inputs, for_D=True)
            
            ## D on real
            D_loss_real = self.gan_criterion(disc_real, torch.ones_like(disc_real))
            clf_loss_real = self.criterion(clf_real, targets)

            ## D on fake
            D_loss_fake = self.gan_criterion(disc_fake, torch.zeros_like(disc_fake))
            clf_loss_fake = self.criterion(clf_fake, torch.ones_like(targets).mul(fake_label).long())

            ## total loss of D
            D_loss = D_loss_real + D_loss_fake + clf_loss_real + clf_loss_fake


            # G
            disc_fake, clf_fake, generated = self.model(noise, inputs, for_D=False)

            ## D on fake
            G_loss_fake = self.gan_criterion(disc_fake, torch.ones_like(disc_fake))
            clf_loss_fake = self.criterion(clf_fake, targets)

            ## total loss of D
            G_loss = G_loss_fake + clf_loss_fake

            mm.track(
                    loss=[
                        (D_loss_real + D_loss_fake).detach().cpu().item(), 
                        G_loss_fake.detach().cpu().item()
                    ], 
                    pred=generated.detach().cpu(),
                    true=inputs.detach().cpu(), 
                    suffix=['D', 'G'],
                    calc_fid=False
                )
            mm.track(clf_loss_real.detach().cpu().item(), clf_real.detach().cpu(), targets)
        else:
            self.model.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results

    @torch.no_grad()
    def evaluate_classifier(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            
            outputs = self.model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)

            mm.track(loss.item(), outputs.detach().cpu(), targets.detach().cpu())
        else:
            self.model.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results