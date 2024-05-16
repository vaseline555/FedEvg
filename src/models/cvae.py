import math
import torch



class CVAE(torch.nn.Module):
    def __init__(self, resize, in_channels, hidden_size, num_classes):
        super(CVAE, self).__init__()
        
    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def sample_z(self, num_samples, dist, width=(-1, 1)):
        """Sample latent vectors"""
        if dist == "mvn":  # multivariate normal
            z = self.mvn_dist.sample((num_samples,))
        elif dist == "truncnorm":  # truncated multivariate normal
            truncnorm_tensor = torch.FloatTensor(
                truncnorm.rvs(a=width[0], b=width[1], size=num_samples * self.z_dim)
            )
            z = torch.reshape(truncnorm_tensor, (num_samples, self.z_dim))
        elif dist == "uniform":  # uniform
            z = torch.FloatTensor(num_samples, self.z_dim).uniform_(*width)

        else:
            raise NotImplementedError(
                "Only multivariate normal (mvn), truncated multivariate normal (truncnorm), and uniform (uniform) distributions supported."
            )

        return z

    def forward(self, X, y_hot, device):
        distributions = self.encoder(X)

        mu = distributions[:, : self.z_dim]
        logvar = distributions[:, self.z_dim :]

        # Re-paramaterization trick, sample latent vector z
        z = self.reparametrize(mu, logvar).to(device)

        # Decode latent vector + class info into a reconstructed image
        x_recon = self.decoder(z, y_hot)

        return x_recon, mu, logvar