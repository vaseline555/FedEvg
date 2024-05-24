import copy
import torch
import logging
import concurrent.futures
import numpy as np

from PIL import Image
from collections import ChainMap, defaultdict

from src import MetricManager, TqdmToLogger
from src.models import ResNet10
from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class FedcvaeServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FedcvaeServer, self).__init__(**kwargs)
        # init server-side classifier and decoder
        classifier = ResNet10(self.args.resize, self.args.in_channels, self.args.hidden_size, self.args.num_classes)
        self.classifier = self._init_model(classifier)
        self.decoder = self.global_model.decoder
        self.latent_dim = self.global_model.latent_dim
        self.results['model_parameter_counts'] = sum(p.numel() for p in self.decoder.parameters()) 

        # init container for storing local decoders and label distributions
        self.local_decoder_container = dict()
        self.local_label_container = dict()

        # ...others
        self.num_train_samples = self.args.num_classes * self.args.spc
        self.aggregated_synthetic_dataset = None # aggregated synthetic samples from local decoders for training global decoder
        self.central_synthetic_dataset = None # synthetic samples from the trained central decoder for training global classifier

    def _log_results(self, resulting_sizes, results, eval, participated, save_raw):
        losses, losses_kl, losses_recon, metrics, num_samples = list(), list(), list(), defaultdict(list), list()
        generated = list()

        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [CLIENT] < {str(identifier).zfill(6)} > '
            if eval: # get loss and metrics
                # loss recon
                loss_recon = result['loss_recon']
                client_log_string += f'| loss (recon.): {loss_recon:.4f} '
                losses_recon.append(loss_recon)

                # loss kl
                loss_kl = result['loss_kl']
                client_log_string += f'| loss (kl): {loss_kl:.4f} '
                losses_kl.append(loss_kl)

                # collect generated samples
                gen_img = result['generated'].unsqueeze(0)
                generated.append(gen_img)

                # metrics
                for metric, value in result['metrics'].items():
                    client_log_string += f'| {metric}: {value:.4f} '
                    metrics[metric].append(value)
            else: # same, but retireve results of last epoch's
                # loss recon
                loss_recon = result[self.args.E]['loss_recon']
                client_log_string += f'| loss (recon): {loss_recon:.4f} '
                losses_recon.append(loss_recon)

                # loss kl
                loss_kl = result[self.args.E]['loss_kl']
                client_log_string += f'| loss (kl): {loss_kl:.4f} '
                losses_kl.append(loss_kl)

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
        losses_recon_array = np.array(losses_recon).astype(float)
        weighted_recon = losses_recon_array.dot(num_samples) / sum(num_samples); std_recon = losses_recon_array.std()

        losses_kl_array = np.array(losses_kl).astype(float)
        weighted_kl = losses_kl_array.dot(num_samples) / sum(num_samples); std_kl = losses_kl_array.std()

        total_log_string += f'\n    - Loss (Recon.): Avg. ({weighted_recon:.4f}) Std. ({std_recon:.4f}) | Loss (KL Div.): Avg. ({weighted_kl:.4f}) Std. ({std_kl:.4f})'
        result_dict['loss'] = {
            'avg_recon': weighted_recon.astype(float), 'std_recon': std_recon.astype(float),
            'avg_kl': weighted_kl.astype(float), 'std_kl': std_kl.astype(float)
        }

        if save_raw:
            result_dict['loss']['raw'] = losses

        self.writer.add_scalars(
            f'Local {"Test" if eval else "Training"} Loss' + eval * f' ({"In" if participated else "Out"})',
            {
                'Avg. (Recon.)': weighted_recon, 'Std. (Recon.)': std_recon,
                'Avg. (KL Div.)': weighted_kl, 'Std. (KL Div.)': std_kl
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
        if len(gen_imgs) > 1:
            viz_idx = np.random.randint(0, len(gen_imgs), size=(), dtype=int)
            gen_imgs = (gen_imgs[viz_idx] * 255).astype(np.uint8)
        gen_imgs = np.transpose(gen_imgs, (1, 2, 0))
        self.writer.add_image(
            f'Local {"Test" if eval else "Training"} Generated Image' + eval * f' ({"In" if participated else "Out"})', 
            gen_imgs, 
            self.round,
            dataformats='HWC'
        )

        # log total message
        self.writer.flush()
        logger.info(total_log_string)
        return result_dict
    
    def _aggregate(self, ids, updated_sizes):
        assert set(updated_sizes.keys()) == set(ids)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')

        # accumulate weights
        for identifier in ids:
            self.local_decoder_container[identifier] = self.clients[identifier].model.decoder
            self.local_label_container[identifier] = self.clients[identifier].compute_local_label_dist()
            self.clients[identifier].model = None
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')
    
    @torch.no_grad()
    def _central_evaluate(self):
        mm = MetricManager(self.args.eval_metrics)
        self.decoder.to(self.args.device)
        self.classifier.to(self.args.device)

        clf_losses, corrects = 0, 0
        for (synth, _, _), (inputs, targets) in zip(
                torch.utils.data.DataLoader(
                    self.aggregated_synthetic_dataset,
                    batch_size=self.args.B,
                    shuffle=False
                ), 
                torch.utils.data.DataLoader(
                    dataset=self.server_dataset, 
                    batch_size=self.args.B, 
                    shuffle=False
                )
            ):
            # real image and label
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            logits = self.classifier(inputs)

            # Calculate losses
            clf_loss = torch.nn.CrossEntropyLoss()(logits, targets)
            clf_losses += clf_loss.item()
            corrects += logits.argmax(1).eq(targets).sum().item()

            # collect clf results
            mm.track(
                loss=[clf_loss.detach().cpu(), torch.zeros(1)], 
                pred=synth.detach().cpu(),
                true=inputs.detach().cpu(), 
                suffix=['clf', 'none'],
                calc_fid=True
            )
            mm.track(clf_loss.item(), logits.detach().cpu(), targets.detach().cpu()) 
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
        
        self.writer.add_scalar('Server Test Loss', clf_losses, 1)
        self.writer.add_scalar('Server Test Acc1', corrects, 1)

        # generated images
        gen_imgs = synth.detach().cpu().numpy()
        if len(gen_imgs) > 1:
            viz_idx = np.random.randint(0, len(gen_imgs), size=(), dtype=int)
            gen_imgs = (gen_imgs[viz_idx] * 255).astype(np.uint8)
        gen_imgs = np.transpose(gen_imgs, (1, 2, 0))
        self.writer.add_image(
            'Server Generated Image', 
            gen_imgs, 
            self.round // self.args.eval_every,
            dataformats='HWC'
        )

        # log TensorBoard
        self.writer.add_scalar(f'Server Test Fid', mm.results['metrics']['fid'], self.round // self.args.eval_every)
        self.writer.flush()
        
        result['fid'] = mm.results['metrics']['fid']
        result['loss'] = clf_losses
        result['acc1'] = corrects
        self.results[self.round]['server_evaluated'] = result

        # local evaluate classifier
        self._request_with_model()

    def _request_with_model(self):
        def __evaluate_clients(client):
            client.classifier = copy.deepcopy(self.classifier)
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
        # aggregate local decoders and label distributions
        self._aggregate(selected_ids, updated_sizes) 

        # generate synthetic data
        local_data_stat = dict()
        for idx in range(self.args.K):
            local_data_stat[idx] = len(self.clients[idx])
        total_data = sum(local_data_stat.values())
        
        # divide train samples according to sample ratio
        ratio = np.array(list(local_data_stat.values())) / total_data
        train_samples = (ratio * (self.num_train_samples - self.args.K)).astype(int) + 1
        
        # replenish remaining samples
        offset = self.num_train_samples - sum(train_samples)
        if offset > 0:
            offset_idx = torch.randint(0, self.args.K, ()).item()
            train_samples[offset_idx] += offset
        
        inputs_synth, latents_synth, targets_synth = [], [], []
        for idx in range(self.args.K):
            # determine number of samples proportional to local sample size
            n_samples = train_samples[idx]

            # init latent inputs and targets following local label distribution
            latents = torch.randn(n_samples, self.latent_dim).to(self.args.device)
            targets = self.local_label_container[idx].multinomial(n_samples, replacement=True)
            targets_hot = torch.eye(self.args.num_classes).to(self.args.device)[
                targets
            ].view(-1, self.args.num_classes)

            # init local decoder
            local_decoder = self.local_decoder_container[idx]
            local_decoder.to(self.args.device)
            local_decoder.eval()

            with torch.no_grad():
                # synthesize samples
                inputs = self.local_decoder_container[idx](torch.cat([latents, targets_hot], dim=1)).detach()
            
            # collect synthesized samples and targets
            inputs_synth.append(inputs.cpu())
            latents_synth.append(latents.cpu())
            targets_synth.append(targets.cpu())
        else:
            self.aggregated_synthetic_dataset = torch.utils.data.dataset.TensorDataset(
                torch.cat(inputs_synth),
                torch.cat(latents_synth),
                torch.cat(targets_synth)
            )

        # wrap into dataloader
        aggregated_synthetic_dataloader = torch.utils.data.DataLoader(
            self.aggregated_synthetic_dataset,
            batch_size=self.args.B,
            shuffle=True
        )

        # train server decoder
        self.decoder.to(self.args.device)
        self.decoder.train()
        kd_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=0.01)
        for epoch in range(self.args.E):
            kd_losses = 0
            for inputs, latents, targets in aggregated_synthetic_dataloader:
                inputs, latents, targets = inputs.to(self.args.device), latents.to(self.args.device), targets.to(self.args.device)
                
                targets_hot = torch.eye(self.args.num_classes).to(self.args.device)[
                    targets
                ].view(-1, self.args.num_classes)
                outputs = self.decoder(torch.cat([latents, targets_hot], dim=1))

                kd_loss = torch.nn.__dict__[self.args.criterion]()(outputs, inputs)
                kd_losses += kd_loss.item() * outputs.size(0)

                kd_optimizer.zero_grad()
                kd_loss.backward()
                kd_optimizer.step()
            kd_losses /= len(self.aggregated_synthetic_dataset)
            logger.info(
                f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [UPDATE] [SERVER] KD Loss: {kd_losses:.4f}'
            )
        self.writer.add_scalar('Server Training KD Loss', kd_losses, 1)
        self.results['server_training_kd_loss'] = kd_losses

        # generate from the trained server decoder
        n_samples = self.num_train_samples
        latents = torch.cat(latents_synth).to(self.args.device) #torch.randn(n_samples, 100).to(self.args.device)
        targets = torch.arange(self.args.num_classes).repeat(n_samples // self.args.num_classes)
        targets_hot = torch.eye(self.args.num_classes).to(self.args.device)[
            targets
        ].view(-1, self.args.num_classes)

        self.decoder.eval()
        with torch.no_grad():
            inputs = self.decoder(torch.cat([latents, targets_hot], dim=1)).cpu()
        self.central_synthetic_dataset = torch.utils.data.TensorDataset(inputs, targets.cpu())
        self.decoder.to('cpu')

        # wrap into dataloader
        central_synthetic_dataloader = torch.utils.data.DataLoader(
            self.central_synthetic_dataset,
            batch_size=self.args.B,
            shuffle=True
        )

        # save server-side synthetic dataset
        inputs_synth = inputs.mul(255).numpy().astype(np.uint8)
        targets_synth = targets.numpy().astype(int)
        np.savez(
            f'{self.args.result_path}/server_generated_{str(self.round).zfill(4)}.npz', 
            inputs=inputs_synth,
            targets=targets_synth
        )

        # train server-side classifier
        self.classifier.to(self.args.device)
        self.classifier.train()
        clf_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        for e in range(self.args.E):
            clf_losses, corrects = 0, 0
            for inputs, targets in central_synthetic_dataloader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                
                outputs = self.classifier(inputs)
                clf_loss = torch.nn.CrossEntropyLoss()(outputs, targets)
                clf_losses += clf_loss.item() * outputs.size(0)
                corrects += outputs.argmax(1).eq(targets).sum().item()

                clf_optimizer.zero_grad()
                clf_loss.backward()
                clf_optimizer.step()
            clf_losses /= len(self.central_synthetic_dataset)
            corrects /= len(self.central_synthetic_dataset)
            logger.info(
                f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [UPDATE] [SERVER] Clf Loss: {clf_losses:.4f} | Acc.: {corrects * 100:.4f}%'
            )
        self.writer.add_scalar('Server Training Loss', clf_losses, 1)
        self.writer.add_scalar('Server Training Acc1', corrects, 1)
        
        self.results['server_training_loss'] = clf_losses
        self.results['server_training_acc1'] = corrects
        return [i for i in range(self.args.K)]
