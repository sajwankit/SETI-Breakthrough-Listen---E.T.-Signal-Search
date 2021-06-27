from torch import nn
from abc import abstractmethod
import timm
import models

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_backbone = models.Backbone(pretrained=pretrained)
        self.fc_mu = nn.Linear(self.encoder_backbone.model.last_linear.out_features, latent_dims)
        self.fc_logvar = nn.Linear(self.encoder_backbone.model.last_linear.out_features, latent_dims)

    def forward(self, x):
        x = self.encoder_backbone(x)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)
        hidden_dims.reverse()

        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        
        self.decoder_backbone = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= in_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
    def forward(self, z):
        z = self.decoder_input(z)
        z = self.decoder_backbone(z)
        z = self.final_layer(z)
        return z

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.kldw =  self.kldw

    def reparameterize(self, mu, logvar):
        '''
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        '''
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.reparameterize(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return [x_recon, latent_mu, latent_logvar]

def vae_loss(recon_x, x, mu, logvar, kldw):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()), dim=1, dim=0)

    loss = recon_loss + kldw * kld_loss
    return [loss, recons_loss, -kld_loss]
    