import torch

from src.models.model_utils import ResidualBlock



__all__ = ['ResNet10', 'ResNet18', 'ResNet34']

CONFIGS = {
    'ResNet10': [1, 1, 1, 1],
    'ResNet18': [2, 2, 2, 2],
    'ResNet34': [3, 4, 6, 3]
}

class ResNet(torch.nn.Module):
    def __init__(self, config, block, resize, in_channels, hidden_size, num_classes, penult_spectral_norm):
        super(ResNet, self).__init__()
        self.hidden_size = hidden_size
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.GroupNorm(hidden_size // 4, hidden_size),
            torch.nn.SiLU() if penult_spectral_norm else torch.nn.ReLU(),
            self._make_layers(block, hidden_size, config[0], stride=1, act=torch.nn.SiLU() if penult_spectral_norm else torch.nn.ReLU()),
            self._make_layers(block, hidden_size * 2, config[1], stride=2, act=torch.nn.SiLU() if penult_spectral_norm else torch.nn.ReLU()),
            self._make_layers(block, hidden_size * 4, config[2], stride=2, act=torch.nn.SiLU() if penult_spectral_norm else torch.nn.ReLU()),
            self._make_layers(block, hidden_size * 8, config[3], stride=2, act=torch.nn.SiLU() if penult_spectral_norm else torch.nn.ReLU()),
            torch.nn.GroupNorm(hidden_size, hidden_size * 8),
            torch.nn.SiLU() if penult_spectral_norm else torch.nn.ReLU(),
            torch.nn.Flatten()
        ) 
        self.classifier = torch.nn.Linear((resize // 8)**2 * (hidden_size * 8), num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _make_layers(self, block, planes, num_blocks, stride, act):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.hidden_size, planes, stride, act))
            self.hidden_size = planes
        return torch.nn.Sequential(*layers)

class ResNet10(ResNet):
    def __init__(self, resize, in_channels, hidden_size, num_classes, penult_spectral_norm=False):
        super(ResNet10, self).__init__(CONFIGS['ResNet10'], ResidualBlock, resize, in_channels, hidden_size, num_classes, penult_spectral_norm)

class ResNet18(ResNet):
    def __init__(self, resize, in_channels, hidden_size, num_classes, penult_spectral_norm=False):
        super(ResNet18, self).__init__(CONFIGS['ResNet18'], ResidualBlock, resize, in_channels, hidden_size, num_classes, penult_spectral_norm)

class ResNet34(ResNet):
    def __init__(self, resize, in_channels, hidden_size, num_classes, penult_spectral_norm=False):
        super(ResNet34, self).__init__(CONFIGS['ResNet34'], ResidualBlock, resize, in_channels, hidden_size, num_classes, penult_spectral_norm)
