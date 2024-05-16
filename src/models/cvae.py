import math
import torch


class Lambda(torch.nn.Module):
    def __init__(self, lam):
        super(Lambda, self).__init__()
        self.lam = lam

    def forward(self, x):
        return self.lam(x)

class Encoder(torch.nn.Module):
    def __init__(self, resize, in_channels, hidden_size, latent_dim):
        super(Encoder, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_size, 4, 2, 1), 
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(hidden_size, hidden_size, 4, 2, 1), 
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1),
            torch.nn.BatchNorm2d(hidden_size * 2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1), 
            torch.nn.BatchNorm2d(hidden_size * 4),
            torch.nn.ReLU(True),
            torch.nn.Flatten(), 
            torch.nn.Linear(hidden_size * 4 * (resize // 16)**2, latent_dim * 2)
        )

    def forward(self, x):
        return self.features(x)

class Decoder(torch.nn.Module):
    def __init__(self, resize, in_channels, hidden_size, num_classes, latent_dim):
        super(Decoder, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + num_classes, hidden_size * 4 * (resize // 16)**2),
            Lambda(lambda x: x.view(-1, hidden_size * 4, resize // 16, resize // 16)),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 4, 2, 1), 
            torch.nn.BatchNorm2d(hidden_size * 2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(hidden_size * 2, hidden_size, 4, 2, 1), 
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1), 
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(hidden_size, in_channels, 4, 2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.features(x)

class CVAE(torch.nn.Module):
    def __init__(self, resize, in_channels, hidden_size, num_classes):
        super(CVAE, self).__init__()
        self.latent_dim = 100
        self.encoder = Encoder(resize, in_channels, hidden_size, latent_dim=100)
        self.decoder = Decoder(resize, in_channels, hidden_size, num_classes, latent_dim=100)

    def _reparametrize(self, mu, std):
        return mu + std * torch.randn_like(mu)

    def forward(self, x, y_hot):
        latent = self.encoder(x)
        mu = latent[:, :self.latent_dim]
        std = torch.nn.functional.softplus(latent[:, self.latent_dim:]).add(1e-8)

        # Re-paramaterization trick, sample latent vector z
        z = self._reparametrize(mu, std)

        # Decode latent vector + class info into a reconstructed image
        generated = self.decoder(torch.cat([z, y_hot], dim=1))
        return generated, mu, std
