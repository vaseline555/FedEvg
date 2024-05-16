import torch
import logging
import numpy as np

from PIL import Image
from collections import defaultdict

from src import MetricManager
from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class FlganServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FlganServer, self).__init__(**kwargs)

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
        gan_criterion = torch.nn.BCELoss()
        self.global_model.to(self.args.device)

        for inputs, targets in torch.utils.data.DataLoader(
            dataset=self.server_dataset, 
            batch_size=self.args.B, 
            shuffle=False
        ):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            fake_label = torch.randint(self.args.num_classes, (inputs.size(0),)).long().to(self.args.device)
            noise = torch.randn(inputs.size(0), self.args.hidden_size * 2, 1, 1).to(self.args.device)
            noise = torch.cat([
                noise, 
                torch.eye(self.args.num_classes).to(self.args.device)[
                    targets
                ].view(-1, self.args.num_classes, 1, 1)
            ], dim=1)

            # D
            disc_fake, disc_real, clf_fake, clf_real = self.global_model(noise, inputs, for_D=True)
            
            ## D on real
            D_loss_real = gan_criterion(disc_real, torch.ones_like(disc_real))
            clf_loss_real = torch.nn.__dict__[self.args.criterion]()(clf_real, targets)

            ## D on fake
            D_loss_fake = gan_criterion(disc_fake, torch.zeros_like(disc_fake))
            clf_loss_fake = torch.nn.__dict__[self.args.criterion]()(clf_fake, torch.ones_like(targets).mul(fake_label).long())

            ## total loss of D
            D_loss = D_loss_real + D_loss_fake + clf_loss_real + clf_loss_fake


            # G
            disc_fake, clf_fake, generated = self.global_model(noise, inputs, for_D=False)

            ## D on fake
            G_loss_fake = gan_criterion(disc_fake, torch.ones_like(disc_fake))
            clf_loss_fake = torch.nn.__dict__[self.args.criterion]()(clf_fake, targets)

            ## total loss of D
            G_loss = G_loss_fake + clf_loss_fake

            mm.track(
                    loss=[(D_loss_real + D_loss_fake).detach().cpu(), G_loss_fake.detach().cpu()], 
                    pred=generated.detach().cpu(),
                    true=inputs.detach().cpu(), 
                    suffix=['D', 'G'],
                    calc_fid=True
                )
            mm.track(clf_loss_real.detach().cpu(), clf_real.detach().cpu(), targets)
        else:
            self.global_model.to('cpu')
            mm.aggregate(len(self.server_dataset))

        # log result
        result = mm.results
        server_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [SERVER] '

        ## loss
        loss = result['loss']
        server_log_string += f'| loss: {loss:.4f} '
        
        ## metrics
        for metric, value in result['metrics'].items():
            server_log_string += f'| {metric}: {value:.4f} '
        logger.info(server_log_string)

        # log TensorBoard
        self.writer.add_scalar('Server Loss', loss, self.round)
        for name, value in result['metrics'].items():
            self.writer.add_scalar(f'Server {name.title()}', value, self.round)
        else:
            self.writer.flush()
        self.results[self.round]['server_evaluated'] = result
