import torch
import logging
import concurrent.futures
import numpy as np

from PIL import Image
from collections import ChainMap, defaultdict

from src import MetricManager, TqdmToLogger
from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class FeddmServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FeddmServer, self).__init__(**kwargs)
        self.results['model_parameter_counts'] = sum(p.numel() for p in self.global_model.parameters()) 
        self.inputs_synth = None
        self.targets_synth = None

    def _log_results(self, resulting_sizes, results, eval, participated, save_raw):
        losses, losses_feature, losses_logit, metrics, num_samples = list(), list(), list(), defaultdict(list), list()
        generated = list()

        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [CLIENT] < {str(identifier).zfill(6)} > '
            if eval:
                loss = result['loss']
                client_log_string += f'| loss : {loss:.4f} '
                losses.append(loss)
                
                for name, value in result['metrics'].items():
                    client_log_string += f'| {name}: {value:.4f} '
                    metrics[name].append(value)          
            else: # get loss and metrics
                # loss logit matching
                loss_logit = result[self.args.E]['loss_logit']
                client_log_string += f'| loss (logit): {loss_logit:.4f} '
                losses_logit.append(loss_logit)

                # loss feature matching
                loss_feature = result[self.args.E]['loss_feature']
                client_log_string += f'| loss (feature): {loss_feature:.4f} '
                losses_feature.append(loss_feature)

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

        if eval:
            # loss
            losses_array = np.array(losses).astype(float)
            weighted_losses = losses_array.dot(num_samples) / sum(num_samples); std_losses = losses_array.std()

            total_log_string += f'\n    - Loss: Avg. ({weighted_losses:.4f}) Std. ({std_losses:.4f})'
            result_dict['loss'] = {'avg_losses': weighted_losses.astype(float), 'std_losses': std_losses.astype(float)}

            if save_raw:
                result_dict['loss']['raw'] = losses

            self.writer.add_scalars(
                f'Local Test Loss ({"In" if participated else "Out"})',
                {'Avg.': weighted_losses, 'Std. (Logit Matching)': std_losses},
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
                    f'Local Test {name.title()} ({"In" if participated else "Out"})',
                    {'Avg.': weighted, 'Std.': std},
                    self.round
                )
        else:
            # loss
            losses_logit_array = np.array(losses_logit).astype(float)
            weighted_logit = losses_logit_array.dot(num_samples) / sum(num_samples); std_logit = losses_logit_array.std()

            losses_feature_array = np.array(losses_feature).astype(float)
            weighted_feature = losses_feature_array.dot(num_samples) / sum(num_samples); std_kl = losses_feature_array.std()

            total_log_string += f'\n    - Loss (Logit Matching): Avg. ({weighted_logit:.4f}) Std. ({std_logit:.4f}) | Loss (Feature Matching): Avg. ({weighted_feature:.4f}) Std. ({std_kl:.4f})'
            result_dict['loss'] = {
                'avg_logit': weighted_logit.astype(float), 'std_logit': std_logit.astype(float),
                'avg_kl': weighted_feature.astype(float), 'std_kl': std_kl.astype(float)
            }

            if save_raw:
                result_dict['loss']['raw'] = losses

            self.writer.add_scalars(
                f'Local {"Test" if eval else "Training"} Loss' + eval * f' ({"In" if participated else "Out"})',
                {
                    'Avg. (Logit Matching)': weighted_logit, 'Std. (Logit Matching)': std_logit,
                    'Avg. (Feature Matching)': weighted_feature, 'Std. (Feature Matching)': std_kl
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
        aggregated_inputs, aggregated_targets = [], []
        for identifier in ids:
            aggregated_inputs.append(self.clients[identifier].inputs_synth)
            aggregated_targets.append(self.clients[identifier].targets_synth)
            self.clients[identifier].model = None
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')
        return torch.cat(aggregated_inputs), torch.cat(aggregated_targets)

    @torch.no_grad()
    def _central_evaluate(self):
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

        # generated images
        gen_imgs = self.inputs_synth.detach().cpu().numpy()
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
        np.savez(
            f'{self.args.result_path}/server_generated_{str(self.round).zfill(4)}.npz', 
            inputs=self.inputs_synth.detach().cpu().numpy(),
            targets=self.targets_synth.detach().cpu().numpy()
        )

        # log TensorBoard
        self.writer.add_scalar(f'Server Test Fid', mm.results['metrics']['fid'], self.round // self.args.eval_every)
        self.writer.flush()
        
        result['fid'] = mm.results['metrics']['fid']
        result['loss'] = clf_losses
        result['acc1'] = corrects
        self.results[self.round]['server_evaluated'] = result

        # local evaluate classifier
        self._request_with_model(self.global_model)

    def _request_with_model(self, model):
        def __evaluate_clients(client):
            if client.model is None:
                client.download(model)
            eval_result = client.evaluate_classifier() 
            client.model = None
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
        self.inputs_synth, self.targets_synth = self._aggregate(selected_ids, updated_sizes) 

        # wrap into dataloader
        aggregated_synthetic_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.inputs_synth, self.targets_synth),
            batch_size=self.args.B,
            shuffle=True
        )

        # train server model
        mm = MetricManager(self.args.eval_metrics)
        self.global_model.to(self.args.device)
        self.global_model.train()

        optimizer = torch.optim.Adam(self.global_model.parameters(), lr=0.001)
        for e in range(self.args.E):
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
        self.global_model.to('cpu')

        self.writer.add_scalar('Server Training Loss', clf_losses, self.round)
        self.writer.add_scalar('Server Training Acc1', corrects, self.round)
        self.results['server_training_loss'] = clf_losses
        self.results['server_training_acc1'] = corrects 
        return selected_ids
