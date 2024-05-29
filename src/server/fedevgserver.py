import os
import copy
import torch
import logging
import torchvision
import concurrent.futures

import numpy as np

from PIL import Image
from collections import ChainMap, defaultdict

from src import MetricManager, TqdmToLogger
from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



def linear_beta_schedule(eta_start=10., eta_end=0., timesteps=100):
    return torch.linspace(eta_start, eta_end, timesteps)

def quadratic_beta_schedule(eta_start=10., eta_end=0., timesteps=100):
    return torch.linspace(eta_start**0.5, eta_end**0.5, timesteps)**2

def sigmoid_beta_schedule(eta_start=10., eta_end=0., timesteps=100):
    etas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(etas) * (eta_end - eta_start) + eta_start

def cosine_beta_schedule(eta_start=10., eta_end=0., timesteps=100, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas * (eta_end - eta_start) + eta_start

class FedevgServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FedevgServer, self).__init__(**kwargs)
        self.results['model_parameter_counts'] = sum(p.numel() for p in self.global_model.parameters()) 
        
        # central buffer
        self.inputs_synth = torch.randn(self.args.num_classes * self.args.spc, self.args.in_channels, self.args.resize, self.args.resize)
        self.targets_synth = torch.arange(self.args.num_classes).view(-1, 1).repeat(1, self.args.spc).view(-1)

        # central SGLD configuration
        self.server_beta_schedule = cosine_beta_schedule(self.args.server_beta, self.args.server_beta_last, self.args.R)
        self.selected_indices = None

    def _log_results(self, resulting_sizes, results, eval, participated, save_raw):
        losses, losses_ce, losses_pcd, metrics, num_samples = list(), list(), list(), defaultdict(list), list()
        generated = list()

        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [CLIENT] < {str(identifier).zfill(6)} > '
            if eval: # get loss and metrics
                # loss
                loss = result['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # loss cross-entropy
                loss_ce = result['loss_ce']
                client_log_string += f'| loss (ce): {loss_ce:.4f} '
                losses_ce.append(loss_ce)

                # loss persistent contrastive divergence
                loss_pcd = result['loss_pcd']
                client_log_string += f'| loss (pcd): {loss_pcd:.4f} '
                losses_pcd.append(loss_pcd)

                # collect generated samples
                gen_img = result['generated'].unsqueeze(0)
                generated.append(gen_img)

                # metrics
                for metric, value in result['metrics'].items():
                    client_log_string += f'| {metric}: {value:.4f} '
                    metrics[metric].append(value)
            else: # same, but retireve results of last epoch's
                # loss
                loss = result[self.args.E]['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # loss cross-entropy
                loss_ce = result[self.args.E]['loss_ce']
                client_log_string += f'| loss (ce): {loss_ce:.4f} '
                losses_ce.append(loss_ce)

                # loss persistent contrastive divergence
                loss_pcd = result[self.args.E]['loss_pcd']
                client_log_string += f'| loss (pcd): {loss_pcd:.4f} '
                losses_pcd.append(loss_pcd)

                # collect generated samples
                gen_img = result[self.args.E]['generated'].unsqueeze(0)
                generated.append(gen_img)

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

        losses_pcd_array = np.array(losses_pcd).astype(float)
        weighted_pcd = losses_pcd_array.dot(num_samples) / sum(num_samples); std_pcd = losses_pcd_array.std()

        losses_ce_array = np.array(losses_ce).astype(float)
        weighted_ce = losses_ce_array.dot(num_samples) / sum(num_samples); std_ce = losses_ce_array.std()

        total_log_string += f'\n    - Loss: Avg. ({weighted:.4f}) Std. ({std:.4f}) | Loss (PCD): Avg. ({weighted_pcd:.4f}) Std. ({std_pcd:.4f}) | Loss (CE): Avg. ({weighted_ce:.4f}) Std. ({std_ce:.4f})'
        result_dict['loss'] = {
            'avg': weighted.astype(float), 'std': std.astype(float),
            'avg_pcd': weighted_pcd.astype(float), 'std_pcd': std_pcd.astype(float),
            'avg_ce': weighted_ce.astype(float), 'std_ce': std_ce.astype(float)
        }

        if save_raw:
            result_dict['loss']['raw'] = losses

        self.writer.add_scalars(
            f'Local {"Test" if eval else "Training"} Loss' + eval * f' ({"In" if participated else "Out"})',
            {
                'Avg.': weighted, 'Std.': std,
                'Avg. (PCD)': weighted_pcd, 'Std. (PCD)': std_pcd,
                'Avg. (CE)': weighted_ce, 'Std. (CE)': std_ce
            },
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

        # generated images
        gen_imgs = np.concatenate(generated)
        viz_idx = np.random.randint(0, len(gen_imgs), size=(), dtype=int)
        to_viz = (gen_imgs[viz_idx] * 255).astype(np.uint8)
        to_viz = np.transpose(to_viz, (1, 2, 0))
        self.writer.add_image(
            f'Local {"Test" if eval else "Training"} Generated Image' + eval * f' ({"In" if participated else "Out"})', 
            to_viz, 
            self.round,
            dataformats='HWC'
        )
        viz_opt = 'L' if to_viz.shape[-1] == 1 else 'RGB'
        img = Image.fromarray(to_viz.squeeze(), viz_opt)
        img.save(f'{self.args.result_path}/{"test" if eval else "train"}_generated_{str(self.round).zfill(4)}.png')
        
        # log total message
        self.writer.flush()
        logger.info(total_log_string)
        return result_dict

    def _request(self, ids, eval, participated, save_raw):
        def __update_clients(client):
            if client.model is None:
                client.download(self.global_model)
            
            # sample from central buffer
            labels = torch.randint(0, self.args.num_classes, (self.args.num_classes * self.args.bpr,))
            indices = torch.randint(0, self.args.spc, (self.args.num_classes * self.args.bpr,))
            self.selected_indices = labels * self.args.spc + indices

            # broadcast buffer
            client.inputs_synth = self.inputs_synth[self.selected_indices].clone()
            client.targets_synth = self.targets_synth[self.selected_indices].clone()

            client.args.lr = self.curr_lr
            update_result = client.update()
            return {client.id: len(client.training_set)}, {client.id: update_result}

        def __evaluate_clients(client, participated):
            if client.model is None:
                assert not participated
                if client.model is None:
                    client.download(self.global_model)
            client.inputs_synth = self.inputs_synth[self.selected_indices].clone()
            client.targets_synth = self.targets_synth[self.selected_indices].clone()

            eval_result = client.evaluate() 
            if not participated:
                client.inputs_synth = None
                client.targets_synth = None
            return {client.id: len(client.test_set)}, {client.id: eval_result}

        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Request {"updates" if not eval else "losses"} to {"all" if ids is None else len(ids)} clients!')
        if eval:
            if self.args.train_only: return
            jobs, results = [], []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() + 4) if self.args.max_workers == -1 else self.args.max_workers) as workhorse:
                for idx in TqdmToLogger(
                    ids, 
                    logger=logger, 
                    desc=f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...evaluate clients... ',
                    total=len(ids)
                    ):
                    jobs.append(workhorse.submit(__evaluate_clients, self.clients[idx], participated))
                for job in concurrent.futures.as_completed(jobs):
                    results.append(job.result())
            _eval_sizes, eval_results = list(map(list, zip(*results)))
            _eval_sizes, eval_results = dict(ChainMap(*_eval_sizes)), dict(ChainMap(*eval_results))
            self.results[self.round][f'clients_evaluated_{"in" if participated else "out"}'] = self._log_results(
                _eval_sizes, 
                eval_results, 
                eval=True, 
                participated=participated,
                save_raw=save_raw
            )
            logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...completed evaluation of {"all" if ids is None else len(ids)} clients!')
            return eval_results
        else:
            jobs, results = [], []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() + 4) if self.args.max_workers == -1 else self.args.max_workers) as workhorse:
                for idx in TqdmToLogger(
                    ids, 
                    logger=logger, 
                    desc=f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...update clients... ',
                    total=len(ids)
                    ):
                    jobs.append(workhorse.submit(__update_clients, self.clients[idx])) 
                for job in concurrent.futures.as_completed(jobs):
                    results.append(job.result())
            update_sizes, _update_results = list(map(list, zip(*results)))
            update_sizes, _update_results = dict(ChainMap(*update_sizes)), dict(ChainMap(*_update_results))
            self.results[self.round]['clients_updated'] = self._log_results(
                update_sizes, 
                _update_results, 
                eval=False, 
                participated=True,
                save_raw=False
            )
            logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...completed updates of {"all" if ids is None else len(ids)} clients!')
            return update_sizes
            
    def _aggregate(self, ids, updated_sizes):
        assert set(updated_sizes.keys()) == set(ids)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')
        
        # calculate mixing coefficients according to sample sizes
        coefficients = {identifier: float(nuemrator / sum(updated_sizes.values())) for identifier, nuemrator in updated_sizes.items()}
        
        # accumulate weights
        e, g = [], []
        for identifier in ids:
            energy_grad, exp_energy_signed = self.clients[identifier].upload()
            self.clients[identifier].inputs_synth = None
            self.clients[identifier].targets_synth = None

            e.append(exp_energy_signed.mul(coefficients[identifier]))
            g.append(energy_grad)
        else:
            energies = torch.stack(e)
            grads = torch.stack(g)
            numerator = (energies[:, :, None, None] * grads).sum(0)
            denominator = energies.sum(0)[:, None, None]
        agg_energy_grad_mixture = numerator.div(denominator)

        # update server-side synthetic data
        sigma = 0.0001
        inputs_synth_curr = self.inputs_synth[self.selected_indices].clone()
        inputs_synth_new = inputs_synth_curr - self.server_beta_schedule[self.round] * agg_energy_grad_mixture + sigma * torch.randn_like(agg_energy_grad_mixture)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')
        return inputs_synth_new.clip(0., 1.)
    
    @torch.no_grad()
    def _central_evaluate(self):
        # plot and save generated images
        self.writer.add_image(
            'Server Generated Test Image', 
            torchvision.utils.make_grid(self.inputs_synth.detach().cpu(), nrow=self.args.spc), 
            self.round // self.args.eval_every
        )

        # save images
        np.savez(
            f'{self.args.result_path}/server_generated_{str(self.round).zfill(4)}.npz', 
            inputs=self.inputs_synth.detach().cpu().numpy(),
            targets=self.targets_synth.detach().cpu().numpy()
        )

        # wrap into dataloader
        aggregated_synthetic_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.inputs_synth, self.targets_synth),
            batch_size=self.args.B,
            shuffle=True
        )

        # train server-side model using synthetic data
        # for measuring generalization performance
        with torch.enable_grad():
            mm = MetricManager(self.args.eval_metrics)
            self.global_model.to(self.args.device)
            self.global_model.train()

            optimizer = torch.optim.Adam(self.global_model.parameters(), lr=0.001)
            clf_losses, corrects = 0, 0
            for inputs, targets in aggregated_synthetic_dataloader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = self.global_model(inputs)

                # Calculate losses
                clf_loss = torch.nn.__dict__[self.args.criterion]()(outputs, targets)
                clf_losses += clf_loss.item()
                corrects += outputs.argmax(1).eq(targets).sum().item()

                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()
                
                # collect clf results
                mm.track(clf_loss.item(), outputs.detach().cpu(), targets.detach().cpu())
            else:
                mm.aggregate(len(self.server_dataset))
                clf_losses /= len(self.server_dataset)
                corrects /= len(self.server_dataset)
                logger.info(
                    f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [UPDATE] [SERVER] Loss: {clf_losses:.4f} Acc.: {corrects * 100:.2f}%'
                )
            self.global_model.eval()
            self.global_model.to('cpu')

        self.writer.add_scalar('Server Training Loss', clf_losses, self.round)
        self.writer.add_scalar('Server Training Acc1', corrects, self.round)
        self.results['server_training_loss'] = clf_losses
        self.results['server_training_acc1'] = corrects 

        # (global) evaluate server model
        mm = MetricManager(self.args.eval_metrics)
        self.global_model.to(self.args.device)

        clf_losses, corrects = 0, 0
        for (synth, _), (inputs, targets) in zip(
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    self.inputs_synth.detach().cpu(), 
                    self.targets_synth.detach().cpu()
                ),
                batch_size=self.args.B,
                shuffle=False
            ), 
            torch.utils.data.DataLoader(
                dataset=self.server_dataset, 
                batch_size=self.args.B, 
                shuffle=False
            )
        ):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            outputs = self.global_model(inputs)

            # Calculate losses
            clf_loss = torch.nn.__dict__[self.args.criterion]()(outputs, targets)
            clf_losses += clf_loss.item()
            corrects += outputs.argmax(1).eq(targets).sum().item()

            # collect for fid
            mm.track(
                loss=[torch.ones(1).mul(-1), torch.ones(1).mul(-1)], 
                pred=synth,
                true=inputs.cpu(), 
                suffix=['none', 'none'],
                calc_fid=True
            )

            # collect clf results
            mm.track(clf_loss.item(), outputs.detach().cpu(), targets.detach().cpu())
        else:
            self.global_model.to('cpu')
            mm.aggregate(len(self.server_dataset))
            clf_losses /= len(self.server_dataset)
            corrects /= len(self.server_dataset)

        # log result
        result = dict()
        server_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [SERVER] '

        ## metrics
        server_log_string += f'| clf loss: {clf_losses:.4f} | accuracy: {corrects * 100:.2f}%'
        logger.info(server_log_string)
        
        self.writer.add_scalar('Server Test Loss', clf_losses, self.round // self.args.eval_every)
        self.writer.add_scalar('Server Test Acc1', corrects, self.round // self.args.eval_every)

        # log TensorBoard
        self.writer.add_scalar(f'Server Test Fid', mm.results['metrics']['fid'], self.round // self.args.eval_every)
        self.writer.flush()
        
        result['fid'] = mm.results['metrics']['fid']
        result['loss'] = clf_losses
        result['acc1'] = corrects
        self.results[self.round]['server_evaluated'] = result

        # (local) evaluate the server model
        self._request_with_model()

    def _request_with_model(self):
        def __evaluate_clients(client):
            client.classifier = copy.deepcopy(self.global_model)
            eval_result = client.evaluate_classifier() 
            client.classifier = None
            return {client.id: len(client.test_set)}, {client.id: eval_result}

        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Request losses to all clients!')
        jobs, results = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.args.K, os.cpu_count() + 4) if self.args.max_workers == -1 else self.args.max_workers) as workhorse:
            for idx in TqdmToLogger(
                [i for i in range(self.args.K)], 
                logger=logger, 
                desc=f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...evaluate clients... ',
                total=self.args.K
                ):
                jobs.append(workhorse.submit(__evaluate_clients, self.clients[idx]))
            for job in concurrent.futures.as_completed(jobs):
                results.append(job.result())
        _eval_sizes, eval_results = list(map(list, zip(*results)))
        _eval_sizes, eval_results = dict(ChainMap(*_eval_sizes)), dict(ChainMap(*eval_results))

        losses, metrics, num_samples = list(), defaultdict(list), list()
        for identifier, result in eval_results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [CLIENT] < {str(identifier).zfill(6)} > '
            # loss
            loss = result['loss']
            client_log_string += f'| loss: {loss:.4f} '
            losses.append(loss)
            
            # metrics
            for metric, value in result['metrics'].items():
                client_log_string += f'| {metric}: {value:.4f} '
                metrics[metric].append(value)
            
            # get sample size
            num_samples.append(_eval_sizes[identifier])

            # log per client
            logger.info(client_log_string)
        else:
            num_samples = np.array(num_samples).astype(float)

        # aggregate into total logs
        result_dict = defaultdict(dict)
        total_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [SUMMARY] ({len(_eval_sizes)} clients):'

        # loss
        losses_array = np.array(losses).astype(float)
        weighted = losses_array.dot(num_samples) / sum(num_samples); std = losses_array.std()

        total_log_string += f'\n    - Loss: Avg. ({weighted:.4f}) Std. ({std:.4f})'
        result_dict['loss'] = {'avg': weighted.astype(float), 'std': std.astype(float),}
        result_dict['loss']['raw'] = losses
        self.writer.add_scalars(
            'Local Test Loss (ALL)',
            {'Avg.': weighted, 'Std.': std},
            self.round
        )

        # metrics
        for name, val in metrics.items():
            val_array = np.array(val).astype(float)
            weighted = val_array.dot(num_samples) / sum(num_samples); std = val_array.std()

            total_log_string += f'\n    - {name.title()}: Avg. ({weighted:.4f}) Std. ({std:.4f})'
            result_dict[name] = {'avg': weighted.astype(float), 'std': std.astype(float)}   
            result_dict[name]['raw'] = val

            self.writer.add_scalars(
                f'Local Test {name.title()} (ALL)',
                {'Avg.': weighted, 'Std.': std},
                self.round
            )

        # log total message
        self.writer.flush()
        logger.info(total_log_string)

        self.results[self.round][f'clients_evaluated_server_model'] = result_dict
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...completed evaluation of all clients!')
        return eval_results
        
    def update(self):
        """Update the global model through federated learning.
        """
        #################
        # Client Update #
        #################
        selected_ids = self._sample_clients() # randomly select clients
        updated_sizes = self._request(selected_ids, eval=False, participated=True, save_raw=False) # request update to selected clients
        _ = self._request(selected_ids, eval=True, participated=True, save_raw=False) # request evaluation to selected clients 
        
        #################
        # Server Update #
        #################
        self.inputs_synth[self.selected_indices] = self._aggregate(selected_ids, updated_sizes) # aggregate local updates
        if self.round % self.args.lr_decay_step == 0: # update learning rate
            self.curr_lr *= self.args.lr_decay

        # plot and save generated images
        self.writer.add_image(
            'Server Generated Training Image', 
            torchvision.utils.make_grid(self.inputs_synth, nrow=self.args.spc), 
            self.round
        )

        """ # WHY?) stateful setting (cross-silo)
        # wrap into dataloader
        aggregated_synthetic_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.inputs_synth, self.targets_synth),
            batch_size=1,
            shuffle=True
        )

        # train server model
        mm = MetricManager(self.args.eval_metrics)
        self.global_model.to(self.args.device)
        self.global_model.train()

        optimizer = torch.optim.SGD(self.global_model.parameters(), lr=0.001, momentum=0.9)
        clf_losses, corrects = 0, 0
        for inputs, targets in aggregated_synthetic_dataloader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            outputs = self.global_model(inputs)

            # Calculate losses
            clf_loss = torch.nn.__dict__[self.args.criterion]()(outputs, targets)
            clf_losses += clf_loss.item()
            corrects += outputs.argmax(1).eq(targets).sum().item()

            optimizer.zero_grad()
            clf_loss.backward()
            optimizer.step()
            
            # collect clf results
            mm.track(clf_loss.item(), outputs.detach().cpu(), targets.detach().cpu())
        else:
            mm.aggregate(len(self.inputs_synth))
            clf_losses /= len(self.inputs_synth)
            corrects /= len(self.inputs_synth)
            logger.info(
                f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [UPDATE] [SERVER] Loss: {clf_losses:.4f} Acc.: {corrects * 100:.2f}%'
            )
        self.global_model.to('cpu')

        self.writer.add_scalar('Server Training Loss', clf_losses, self.round)
        self.writer.add_scalar('Server Training Acc1', corrects, self.round)
        self.results['server_training_loss'] = clf_losses
        self.results['server_training_acc1'] = corrects
        """
        return selected_ids
        