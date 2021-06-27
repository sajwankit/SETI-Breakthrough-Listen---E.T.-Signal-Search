import torch
from torch import nn
from abc import abstractmethod
import timm
import models
import config

from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)


class VAE_loss(nn.Module):
    def __init__(self, kldw=1):
        super().__init__()

        self.kldw = kldw
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def gaussian_likelihood(self, recon_x, log_scale, x):
        dist = torch.distributions.Normal(recon_x, torch.exp(log_scale.to(config.DEVICE)))

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3)).mean()

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu).to(config.DEVICE), torch.ones_like(std).to(config.DEVICE))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl.mean()
    
    def forward(self, recon_x, x, mu, log_var, z):
        recon_loss = self.gaussian_likelihood(recon_x, self.log_scale, x)
        kld_loss = self.kl_divergence(z, mu, std=torch.exp(log_var / 2))
        loss = recon_loss + self.kldw * kld_loss
        return [loss, recon_loss, -kld_loss]

# def decoder_final_layer():
#     self.final_layer = nn.Sequential(
#                             nn.ConvTranspose2d(hidden_dims[-1],
#                                                hidden_dims[-1],
#                                                kernel_size=3,
#                                                stride=2,
#                                                padding=1,
#                                                output_padding=1),
#                             nn.BatchNorm2d(hidden_dims[-1]),
#                             nn.LeakyReLU(),
#                             nn.Conv2d(hidden_dims[-1], out_channels= in_channels,
#                                       kernel_size= 3, padding= 1),
#                             nn.Tanh())

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        modules = []
        hidden_dims = [1024, 512, 256, 128, 64, 32, 16, 8]
        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_dims[i],
                                       out_channels=hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder_backbone = nn.Sequential(*modules)
        self.decoder_final_layer = nn.Sequential(
                                            nn.ConvTranspose2d(
                                                                in_channels=hidden_dims[-1],
                                                                out_channels=config.CHANNELS,
                                                                kernel_size=3,
                                                                stride = 2,
                                                                padding=1,
                                                                output_padding=1
                                                                ),
                                            nn.BatchNorm2d(config.CHANNELS),
                                            nn.LeakyReLU(),
                                            )
                                            
    def forward(self, z):
        recon_x = self.decoder_backbone(z)
        recon_x = self.decoder_final_layer(recon_x)
        return recon_x

class VAE(nn.Module):
    def __init__(self, latent_dim=1024, input_shape=(32, 3, 258, 256)):
        super().__init__()

        # self.encoder = resnet18_encoder(first_conv=False, maxpool1=False)
        self.encoder_backbone = models.Backbone(pretrained=True)
        # self.decoder_backbone = resnet18_decoder(
        #     latent_dim=latent_dim,
        #     input_height=256,
        #     first_conv=False,
        #     maxpool1=False
        # )

        self.decoder = Decoder(latent_dim=latent_dim, )

        # distribution parametersInterpolate
        self.fc_mu = nn.Linear(self.encoder_backbone.model.last_linear.out_features, latent_dim)
        self.fc_var = nn.Linear(self.encoder_backbone.model.last_linear.out_features, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        x_encoded = self.encoder_backbone(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        '''
        sample z from q
        '''
        std = torch.exp(log_var/2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        '''
        reconstructed x from decoder
        '''
        z = z.view(z.size(0), z.size(1), 1, 1)
        recon_x = self.decoder(z)
        recon_x = nn.functional.interpolate(recon_x, size=(258, 256))
        return recon_x, x, mu, log_var, z
    
    
# def vae_loss(recon_x, x, mu, logvar, kldw):
#     recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
#     kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()), dim=1, dim=0)

#     loss = recon_loss + kldw * kld_loss
#     return [loss, recons_loss, -kld_loss]