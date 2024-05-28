import math
import torch



class Lambda(torch.nn.Module):
    def __init__(self, lam):
        super(Lambda, self).__init__()
        self.lam = lam

    def forward(self, x):
        return self.lam(x)

class Generator(torch.nn.Module):
    def __init__(self, resize, in_channels, hidden_size, num_classes):
        super(Generator, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(hidden_size * 2 + num_classes, hidden_size * 4, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(hidden_size * 4),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(hidden_size * 2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(hidden_size * 2, hidden_size, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(hidden_size, in_channels, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, resize, in_channels, hidden_size, num_classes):
        super(Discriminator, self).__init__()
        proj_size = resize // 2**3
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_size, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(hidden_size * 2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(hidden_size * 4),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(hidden_size * 4, hidden_size * 4, 4, 1, 0, bias=False),
            torch.nn.Dropout(0.5),
            torch.nn.Flatten(),
            torch.nn.Linear(hidden_size * 4, 1 + num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        gen_disc, cls_disc = x[:, :1], x[:, 1:]
        return gen_disc, cls_disc

class ACGAN(torch.nn.Module):
    def __init__(self, resize, in_channels, hidden_size, num_classes):
        super(ACGAN, self).__init__()
        self.generator = Generator(resize, in_channels, hidden_size, num_classes)
        self.discriminator = Discriminator(resize, in_channels, hidden_size, num_classes)

    def forward(self, noise, real, for_D=True):
        x_fake = self.generator(noise)
        if for_D:
            disc_res, clf_res = self.discriminator(
                torch.cat([x_fake.detach(), real], dim=0)
            )
            disc_fake, disc_real = disc_res[:x_fake.size(0)], disc_res[x_fake.size(0):]
            clf_fake, clf_real = clf_res[:x_fake.size(0)], clf_res[x_fake.size(0):]
            return disc_fake, disc_real, clf_fake, clf_real
        else:
            disc_fake, clf_fake = self.discriminator(x_fake)
            return disc_fake, clf_fake, x_fake.detach()
