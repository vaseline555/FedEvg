import copy
import torch
import logging
import torchvision
import concurrent.futures
import numpy as np

from PIL import Image
from collections import ChainMap, defaultdict

from src import MetricManager, TqdmToLogger
from src.models import ResNet10
from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class FedcganServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FedcganServer, self).__init__(**kwargs)
        classifier = ResNet10(self.args.resize, self.args.in_channels, self.args.hidden_size, self.args.num_classes)
        self.classifier = self._init_model(classifier)
        self.results['model_parameter_counts'] = sum(p.numel() for p in self.global_model.parameters()) 

    def _log_results(self, resulting_sizes, results, eval, participated, save_raw):
        losses, losses_D, losses_G, metrics, num_samples = list(), list(), list(), defaultdict(list), list()
        generated = list()

        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [CLIENT] < {str(identifier).zfill(6)} > '
            if eval: # get loss and metrics
                # loss
                loss = result['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # loss G
                loss_G = result['loss_G']
                client_log_string += f'| loss (G): {loss_G:.4f} '
                losses_G.append(loss_G)

                # loss D
                loss_D = result['loss_D']
                client_log_string += f'| loss (D): {loss_D:.4f} '
                losses_D.append(loss_D)

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
                
                # loss G
                loss_G = result[self.args.E]['loss_G']
                client_log_string += f'| loss (G): {loss_G:.4f} '
                losses_G.append(loss_G)

                # loss D
                loss_D = result[self.args.E]['loss_D']
                client_log_string += f'| loss (D): {loss_D:.4f} '
                losses_D.append(loss_D)

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

        losses_G_array = np.array(losses_G).astype(float)
        weighted_G = losses_G_array.dot(num_samples) / sum(num_samples); std_G = losses_G_array.std()

        losses_D_array = np.array(losses_D).astype(float)
        weighted_D = losses_D_array.dot(num_samples) / sum(num_samples); std_D = losses_D_array.std()

        total_log_string += f'\n    - Loss: Avg. ({weighted:.4f}) Std. ({std:.4f}) | Loss (G): Avg. ({weighted_G:.4f}) Std. ({std_G:.4f}) | Loss (D): Avg. ({weighted_D:.4f}) Std. ({std_D:.4f})'
        result_dict['loss'] = {
            'avg': weighted.astype(float), 'std': std.astype(float),
            'avg_g': weighted_G.astype(float), 'std_g': std_G.astype(float),
            'avg_d': weighted_D.astype(float), 'std_d': std_D.astype(float)
        }

        if save_raw:
            result_dict['loss']['raw'] = losses

        self.writer.add_scalars(
            f'Local {"Test" if eval else "Training"} Loss' + eval * f' ({"In" if participated else "Out"})',
            {
                'Avg.': weighted, 'Std.': std,
                'Avg. (G)': weighted_G, 'Std. (G)': std_G,
                'Avg. (D)': weighted_D, 'Std. (D)': std_D
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

    @torch.no_grad()
    def _central_evaluate(self):
        mm = MetricManager(self.args.eval_metrics)

        #######################
        # 1. Generate Samples #
        #######################
        # generate server-side synthetic samples
        num_samples = self.args.num_classes * self.args.spc
        inputs_synth, targets_synth = [], []
        
        self.global_model.to(self.args.device)
        self.global_model.eval()
        with torch.no_grad():
            targets_synth = torch.arange(self.args.num_classes).view(-1, 1).repeat(1, self.args.spc).view(-1).to(self.args.device)
            noise = torch.rand(num_samples, self.args.hidden_size * 2, 1, 1).sub(0.5).div(0.5).to(self.args.device)
            noise = torch.cat([
                noise, 
                torch.eye(self.args.num_classes).to(self.args.device)[
                    targets_synth
                ].view(-1, self.args.num_classes, 1, 1)
            ], dim=1)
            inputs_synth = self.global_model.generator(noise)
        self.global_model.to('cpu')
        
        # log generated images    
        targets_synth, sorted_indices = torch.sort(targets_synth.detach().cpu(), 0)
        inputs_synth = inputs_synth[sorted_indices].detach().cpu()

        self.writer.add_image(
            'Server Generated Test Image', 
            torchvision.utils.make_grid(inputs_synth, nrow=self.args.spc), 
            self.round // self.args.eval_every
        )

        # save server-side synthetic dataset
        np.savez(
            f'{self.args.result_path}/server_generated_{str(self.round).zfill(4)}.npz', 
            inputs=inputs_synth.mul(0.5).add(0.5).numpy(),
            targets=targets_synth.numpy().astype(int)
        )

        #######################
        # 2. Train Classifier #
        #######################
        # train server-side classifier
        self.classifier.to(self.args.device)
        self.classifier.train()
        with torch.enable_grad():
            clf_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.01)
            for (inputs, targets) in torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                        inputs_synth, 
                        targets_synth
                    ),
                    batch_size=self.args.B,
                    shuffle=True
                ):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = self.classifier(inputs)

                # Calculate losses
                clf_loss = torch.nn.CrossEntropyLoss()(outputs, targets)
                clf_losses += clf_loss.item()
                corrects += outputs.argmax(1).eq(targets).sum().item()

                clf_optimizer.zero_grad()
                clf_loss.backward()
                clf_optimizer.step()

                # collect clf results
                mm.track(clf_loss.item(), outputs.detach().cpu(), targets.detach().cpu())
            else:
                self.classifier.to('cpu')
                mm.aggregate(len(self.server_dataset))
                clf_losses /= len(self.server_dataset)
                corrects /= len(self.server_dataset)

        # log result
        result = dict()
        server_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [SERVER] '

        ## metrics
        server_log_string += f'| clf loss: {clf_losses:.4f} | accuracy: {corrects * 100:.2f}%'
        logger.info(server_log_string)
        
        self.writer.add_scalar('Server Training Loss', clf_losses, self.round // self.args.eval_every)
        self.writer.add_scalar('Server Training Acc1', corrects * 100, self.round // self.args.eval_every)

        ##########################
        # 3. Evaluate Classifier #
        ##########################
        # evaluate server-side classifier
        mm = MetricManager(self.args.eval_metrics)
        self.classifier.to(self.args.device)
        self.classifier.eval()
        clf_losses, corrects = 0, 0
        for (synth, _), (inputs, targets) in zip(
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(inputs_synth, targets_synth),
                batch_size=self.args.B,
                shuffle=False
            ), 
            torch.utils.data.DataLoader(
                dataset=self.server_dataset, 
                batch_size=self.args.B, 
                shuffle=False
            )
        ):
            inputs, targets = inputs.sub(0.5).div(0.5).to(self.args.device), targets.to(self.args.device)
            outputs = self.classifier(inputs)

            # Calculate losses
            clf_loss = torch.nn.CrossEntropyLoss()(outputs, targets)
            clf_losses += clf_loss.item()
            corrects += outputs.argmax(1).eq(targets).sum().item()

            # collect for fid
            mm.track(
                loss=[torch.ones(1).mul(-1).item(), torch.ones(1).mul(-1).item()], 
                pred=synth.mul(0.5).add(0.5).cpu(),
                true=inputs.mul(0.5).add(0.5).cpu(), 
                suffix=['none', 'none'],
                calc_fid=True
            )

            # collect clf results
            mm.track(clf_loss.item(), outputs.detach().cpu(), targets.detach().cpu())
        else:
            self.classifier.to('cpu')
            mm.aggregate(len(self.server_dataset))
            clf_losses /= len(self.server_dataset)
            corrects /= len(self.server_dataset)

        # log result
        result = mm.results
        del result['generated']
        server_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [SERVER] '

        # loss
        server_log_string += f'| clf loss: {clf_losses:.4f} | accuracy: {corrects * 100:.2f}%'
        logger.info(server_log_string)
        
        # log TensorBoard
        self.writer.add_scalar('Server Test Loss', clf_losses, self.round // self.args.eval_every)
        for name, value in result['metrics'].items():
            self.writer.add_scalar(f'Server Test {name.title()}', value, self.round // self.args.eval_every)
        else:
            self.writer.flush()
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