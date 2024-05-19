import torch
import warnings

import scipy as sp
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve,\
    average_precision_score, f1_score, precision_score, recall_score,\
        mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,\
            r2_score, d2_pinball_score, top_k_accuracy_score, balanced_accuracy_score

from src.models import InceptionV3
from .basemetric import BaseMetric

warnings.filterwarnings('ignore')



class Acc1(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else: 
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores > cutoff, 1, 0)
        return accuracy_score(answers, labels)

class Acc5(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).softmax(-1).numpy()
        answers = torch.cat(self.answers).numpy()
        num_classes = scores.shape[-1]
        return top_k_accuracy_score(answers, scores, k=5, labels=np.arange(num_classes))

class Auroc(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).sigmoid().numpy()
        answers = torch.cat(self.answers).numpy()
        num_classes = scores.shape[-1]
        return roc_auc_score(answers, scores, average='weighted', multi_class='ovr', labels=np.arange(num_classes))

class Auprc(BaseMetric): # only for binary classification
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).sigmoid().numpy()
        answers = torch.cat(self.answers).numpy()
        return average_precision_score(answers, scores, average='weighted')

class Youdenj(BaseMetric):  # only for binary classification
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).sigmoid().numpy()
        answers = torch.cat(self.answers).numpy()
        fpr, tpr, thresholds = roc_curve(answers, scores)
        return float(thresholds[np.argmax(tpr - fpr)])

class F1(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else: 
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return f1_score(answers, labels, average='weighted', zero_division=0)

class Precision(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else: 
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return precision_score(answers, labels, average='weighted', zero_division=0)

class Recall(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else: 
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return recall_score(answers, labels, average='weighted', zero_division=0)

class Seqacc(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        num_classes = pred.size(-1)
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p.view(-1, num_classes))
        self.answers.append(t.view(-1))

    def summarize(self):
        labels = torch.cat(self.scores).argmax(-1).numpy()
        answers = torch.cat(self.answers).numpy()

        # ignore special tokens
        labels = labels[answers != -1]
        answers = answers[answers != -1]
        return np.nan_to_num(accuracy_score(answers, labels))

class Mse(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return mean_squared_error(answers, scores)

class Rmse(Mse):
    def __init__(self):
        super(Rmse, self).__init__()

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return mean_squared_error(answers, scores, squared=False)

class Mae(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return mean_absolute_error(answers, scores)

class Mape(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return mean_absolute_percentage_error(answers, scores)

class R2(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self, *args):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return r2_score(answers, scores)

class D2(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self, *args):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return d2_pinball_score(answers, scores)

class Dice(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self, epsilon=1e-6, *args):
        SPATIAL_DIMENSIONS = 2, 3, 4
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers)
        tp = scores.mul(answers).sum(dim=SPATIAL_DIMENSIONS)
        fp = scores.mul(1 - answers).sum(dim=SPATIAL_DIMENSIONS)
        fn = (1 - scores).mul(answers).sum(dim=SPATIAL_DIMENSIONS)
        dice = (tp.mul(2)).div(tp.mul(2).add(fp.add(fn).add(epsilon))).mean()
        return torch.nan_to_num(dice, 0.).item()
        
class Balacc(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(dim=1).numpy()
        else: 
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores > cutoff, 1, 0)
        return balanced_accuracy_score(answers, labels)

class Fid(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
    
    @torch.no_grad()
    def _calculate_activation_statistics(self, images, batch_size=32, dims=2048, cuda=True):
        device = f'cuda:{torch.randint(0, torch.cuda.device_count(), ()).item()}' if cuda else 'cpu'

        inception = InceptionV3(normalize_input=False)
        inception.to(device)
        inception.eval()

        # repeat channels for gray images
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
    
        acts = []
        for image in torch.utils.data.DataLoader(images, batch_size=batch_size):
            image = image.to(device)
            pred = inception(image)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))

            act = pred.cpu().reshape(pred.size(0), -1)
            acts.append(act)
        act = torch.cat(acts).numpy()
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2
        covmean, _ = sp.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sp.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t.sub(0.5).div(0.5))

    def summarize(self):
        try:
            scores = torch.cat(self.scores)
            answers = torch.cat(self.answers)
        except:
            return -1

        mu_1, std_1 = self._calculate_activation_statistics(answers, cuda=True)
        mu_2, std_2 = self._calculate_activation_statistics(scores, cuda=True)
        return self._calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
