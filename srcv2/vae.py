import torch
from torch import nn
from abc import abstractmethod
import timm
import models
import config
import math
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
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
        return log_pxz.sum(dim=(1, 2, 3))

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
        # kl = 0 means both distributions are identical
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    
    def forward(self, recon_x, x, mu, log_var, z):
        recon_loss = self.gaussian_likelihood(recon_x, self.log_scale, x)
        kld_loss = self.kl_divergence(z, mu, std=torch.exp(log_var / 2))
# #         scale = pow(10, len(str(int(recon_loss.mean())).replace('-', '')) - len(str(int(kld_loss.mean())).replace('-', '')))
        #elbo loss
        elbo = self.kldw*kld_loss - recon_loss
#         recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
#         kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

#         loss = recon_loss + self.kldw * kld_loss
        loss = elbo 
        return [loss.mean(), recon_loss.mean(), kld_loss.mean()]

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

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        modules = []
        hidden_dims = [1024, 512, 256, 128, 64, 32, 16, 8]
        hidden_dims.reverse()
        print(hidden_dims)
        self.encoder_init_layer = nn.Sequential(
                                            nn.Conv2d(
                                                                in_channels=3,
                                                                out_channels=hidden_dims[0],
                                                                kernel_size=3,
                                                                stride = 2,
                                                                padding=1,
                                                                ),
                                            nn.BatchNorm2d(hidden_dims[0]),
                                            nn.LeakyReLU(),
                                            )
        
        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=hidden_dims[i],
                                       out_channels=hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                             ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.encoder_backbone = nn.Sequential(*modules)

    def forward(self, x):
        x = self.encoder_init_layer(x)
        x = self.encoder_backbone(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        modules = []
        hidden_dims = [1024*4, 1024, 512, 128, 32, 16, 8]
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
                    
                )
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
                                            
                                            )
                                            
    def forward(self, z):
        recon_x = self.decoder_backbone(z)
        recon_x = self.decoder_final_layer(recon_x)
        return recon_x

class AE(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()

        # self.encoder = resnet18_encoder(first_conv=False, maxpool1=False)
        self.encoder = Encoder(latent_dim=latent_dim,)

        self.decoder = Decoder(latent_dim=latent_dim, )

        # distribution parametersInterpolate
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        xt = nn.functional.interpolate(x, size=(256, 256))
#         xt = xt.repeat(1, 3, 1, 1)

        x_encoded = self.encoder(xt)
#         mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

#         '''
#         sample z from q
#         '''
#         std = torch.exp(log_var/2)
#         q = torch.distributions.Normal(mu, std)
#         z = q.rsample()

#         '''
#         reconstructed x from decoder
#         '''
#         z_temp = z.view(z.size(0), z.size(1), 1, 1)
        recon_x = self.decoder(x_encoded)
        recon_x = nn.functional.interpolate(recon_x, size=(config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
        recon_x = nn.functional.instance_norm(recon_x)
#         recon_x = recon_x[:, 0, :, :]
        return recon_x, x, torch.randn(1), torch.randn(1), torch.randn(1)
    

######################################################################################################################    
class AE2(nn.Module):
    def __init__(self, latent_dim=1024*4):
        super().__init__()

        # self.encoder = resnet18_encoder(first_conv=False, maxpool1=False)
        self.encoder_backbone = models.Backbone(pretrained=True)
        self.encoder_output = nn.Linear(1000, 1024*4)
        self.decoder = Decoder(latent_dim=latent_dim, )

        # distribution parametersInterpolate
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        xt = nn.functional.interpolate(x, size=(256, 256))
#         xt = xt.repeat(1, 3, 1, 1)

        x_encoded = self.encoder_backbone(xt)
        x_encoded = self.encoder_output(x_encoded)
        x_encoded = x_encoded.view(x_encoded.size(0), x_encoded.size(1), 1, 1)
#         mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

#         '''
#         sample z from q
#         '''
#         std = torch.exp(log_var/2)
#         q = torch.distributions.Normal(mu, std)
#         z = q.rsample()

#         '''
#         reconstructed x from decoder
#         '''
#         z_temp = z.view(z.size(0), z.size(1), 1, 1)
        recon_x = self.decoder(x_encoded)
        recon_x = nn.functional.interpolate(recon_x, size=(config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
        recon_x = nn.functional.instance_norm(recon_x)
#         recon_x = recon_x[:, 0, :, :]
        return recon_x, x, torch.randn(1), torch.randn(1), torch.randn(1)    
    
    
    
    
    
    
    
    

# def vae_loss(recon_x, x, mu, logvar, kldw):
#     recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
#     kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()), dim=1, dim=0)

#     loss = recon_loss + kldw * kld_loss
#     return [loss, recons_loss, -kld_loss]
######################################################################################################

# class VAE(nn.Module):
#     def __init__(self, latent_dim=1024, input_shape=(32, 3, 273, 256)):
#         super().__init__()

#         # self.encoder = resnet18_encoder(first_conv=False, maxpool1=False)
#         self.encoder_backbone = models.Backbone(pretrained=True)
#         # self.decoder_backbone = resnet18_decoder(
#         #     latent_dim=latent_dim,
#         #     input_height=256,
#         #     first_conv=False,
#         #     maxpool1=False
#         # )

#         self.decoder = Decoder(latent_dim=latent_dim, )

#         # distribution parametersInterpolate
#         self.fc_mu = nn.Linear(self.encoder_backbone.model.last_linear.out_features, latent_dim)
#         self.fc_var = nn.Linear(self.encoder_backbone.model.last_linear.out_features, latent_dim)

#         # for the gaussian likelihood
#         self.log_scale = nn.Parameter(torch.Tensor([0.0]))

#     def forward(self, x):
#         x_encoded = self.encoder_backbone(x)
#         mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

#         '''
#         sample z from q
#         '''
#         std = torch.exp(log_var/2)
#         q = torch.distributions.Normal(mu, std)
#         z = q.rsample()

#         '''
#         reconstructed x from decoder
#         '''
#         z_temp = z.view(z.size(0), z.size(1), 1, 1)
#         recon_x = self.decoder(z_temp)
#         recon_x = nn.functional.interpolate(recon_x, size=(config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
#         recon_x = nn.functional.instance_norm(recon_x)
#         return recon_x, x, mu, log_var, z
    
    
# def vae_loss(recon_x, x, mu, logvar, kldw):
#     recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
#     kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()), dim=1, dim=0)

#     loss = recon_loss + kldw * kld_loss
#     return [loss, recons_loss, -kld_loss]



class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=1024, input_height=256):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
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

    def forward(self, x):
        xt = nn.functional.interpolate(x, size=(256, 256))
        xt = xt.repeat(1, 3, 1, 1)
        x_encoded = self.encoder(xt)
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
#         z_temp = z.view(z.size(0), z.size(1), 1, 1)
        recon_x = self.decoder(z)
        recon_x = nn.functional.interpolate(recon_x, size=(config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
        recon_x = nn.functional.instance_norm(recon_x)
        return recon_x, x, mu, log_var, z

######################################################################################################
class BetaVAE(nn.Module):
    num_iter = 0
    def __init__(self,
                 latent_dim=1024,
                 input_shape=(32, 3, 273, 256),
                ):
        super().__init__()


        self.encoder_backbone = models.Backbone(pretrained=True)

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
        z_temp = z.view(z.size(0), z.size(1), 1, 1)
        recon_x = self.decoder(z_temp)
        recon_x = nn.functional.interpolate(recon_x, size=(273, 256))
        recon_x = nn.functional.instance_norm(recon_x)
        return recon_x, x, mu, log_var, z

    
class BetaVAE_loss(nn.Module):
    def __init__(self,
             kldw=1,
             beta=4,
             gamma=1000,
             max_capacity=25,
             capacity_max_iter=1e5,
             loss_type='B'
            ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.c_max = torch.Tensor([max_capacity])
        self.c_stop_iter = capacity_max_iter
        self.num_iter = 0
        self.kldw = kldw
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def gaussian_likelihood(self, recon_x, log_scale, x):
        dist = torch.distributions.Normal(recon_x, torch.exp(log_scale.to(config.DEVICE)))

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

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
        # kl = 0 means both distributions are identical
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    
    def forward(self, recon_x, x, mu, log_var, z):
        self.num_iter += 1
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        self.c_max = self.c_max.to(config.DEVICE)
        c = torch.clamp(self.c_max/self.c_stop_iter*self.num_iter, 0, self.c_max.data[0])
        loss = recon_loss + self.gamma*self.kldw * (kld_loss-c).abs()
        
        return [loss.mean(), recon_loss.mean(), kld_loss.mean()]