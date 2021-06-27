from torch import nn
from abc import abstractmethod
import timm
import models
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)

def vae_loss(recon_x, x, mu, logvar, kldw):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()), dim=1, dim=0)

    loss = recon_loss + kldw * kld_loss
    return [loss, recons_loss, -kld_loss]


class VAE(nn.Module):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_shape=(256, 258)):
        super().__init__()

        self.encoder = resnet18_encoder9(first_conv=False, maxpool1=False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )
        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

     def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def forward(self, x):
        x_encoded = self.encoder(x)
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
        recon_x = self.decoder(z)

        return recon_x, x, mu, log_var
