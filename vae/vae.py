from torch import nn
from abc import abstractmethod
import timm
import models

# class BaseVAE(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def encode(self, x):
#         raise NotImplementedError
    
#     def decode(self, z):
#         raise NotImplementedError
    
#     def sample(self, batch_size, current_device, **kwargs):
#         raise RuntimeWarning()
    
#     def generate(self, x, **kwargs):
#         raise NotImplementedError
    
#     def forward(self, x):
#         pass
    
#     def loss_function(self, recon_x, x, mu, logvar):
#         pass

# class CNNVAE(BaseVAE):
#     def __init__(self, kldw=0.5, pretrained=True, in_channels=3, latent_dim=2, hidden_dims=None, **kwargs):
#         super(BaseVAE).__init__()
#         self.laten_dim = latent_dim
#         self.kldw =  self.kldw
#         '''
#         Build Encoder
#         '''
#         self.encoder_backbone = models.Backbone(pretrained=pretrained)
#         self.fc_mu = nn.Linear(self.encoder_backbone.model.last_linear.out_features, latent_dim)
#         self.fc_var = nn.Linear(self.encoder_backbone.model.last_linear.out_features, latent_dim)

#         '''
#         Build Decoder
#         '''
#         modules = []
#         if hidden_dims is None:
#             hidden_dims = [32, 64, 128, 256, 512]

#         self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)
#         hidden_dims.reverse()

#         for i in range(len(hidden_dims)-1):
#             modules.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(hidden_dims[i],
#                                        hidden_dims[i + 1],
#                                        kernel_size=3,
#                                        stride = 2,
#                                        padding=1,
#                                        output_padding=1),
#                     nn.BatchNorm2d(hidden_dims[i + 1]),
#                     nn.LeakyReLU())
#             )
        
#         self.decoder = nn.Sequential(*modules)

#         self.final_layer = nn.Sequential(
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
        
#     def encoder(self, x):
#         x = self.encoder_backbone(x)
#         mu = self.fc_mu(x)
#         logvar = self.fc_var(x)

#         return [mu, logvar]
    
#     def decoder(self, z):
#         '''
#         Maps the given latent vector onto the image space
#         '''
#         z = self.decoder_input(z)
#         z = self.decoder(z)
#         z = self.final_layer(z)
#         return z
    
#     def reparameterize(self, mu, logvar):
#         '''
#         Reparameterization trick to sample from N(mu, var) from
#         N(0,1).
#         :param mu: (Tensor) Mean of the latent Gaussian [B x D]
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
#         :return: (Tensor) [B x D]
#         '''
#         std = logvar.mul(0.5).exp_()
#         eps = torch.empty_like(std).normal_()
#         return eps*std+mu
    
#     def forward(self, x):
#         mu, logvar = self.encoder(x)
#         if self.training:
#             z = self.reparameterize(mu, logvar)
#         else:
#             return mu
    
#     def loss_function(self, recon_x, x, mu, logvar):
#         recon_loss = nn.functional.mse_loss(recon_x, x)
#         kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()), dim=1, dim=0)

#         loss = recon_loss + kld_weight * kld_loss
#         return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

#      def sample(self, z
#                num_samples,
#                device):
#         """
#         Samples from the latent space and return the corresponding
#         image space map.
#         :param num_samples: (Int) Number of samples
#         :param device: (Int) Device to run the model
#         :return: (Tensor)
#         """
#         z = torch.randn(num_samples,
#                         self.latent_dim)

#         z = z.to(device)

#         samples = self.decode(z)
#         return samples

#     def generate(self, x) -> Tensor:
#         """
#         Given an input image x, returns the reconstructed image
#         :param x: (Tensor) [B x C x H x W]
#         :return: (Tensor) [B x C x H x W]
#         """

#         return self.forward(x)[0]


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
        return x_recon, latent_mu, latent_logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()), dim=1, dim=0)

        loss = recon_loss + self.kldw * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
    