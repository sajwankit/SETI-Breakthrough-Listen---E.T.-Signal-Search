from torch import nn
from abc import abstractmethod
import timm
import models

class BaseVAE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def encode(self, input):
        raise NotImplementedError
    
    def decode(self, input):
        raise NotImplementedError
    
    def sample(self, batch_size, current_device, **kwargs):
        raise RuntimeWarning()
    
    def generate(self, x, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, inputs):
        pass
    
    @abstractmethod
    def loss_function(self, inputs):
        pass

class CNNVAE(BaseVAE):
    def __init__(self, pretrained=True, in_channels=3, latent_dim=2, hidden_dims=None, **kwargs):
        super(BaseVAE).__init__()
        self.laten_dim = latent_dim
        
        '''
        Build Encoder
        '''
        self.encoder_backbone = models.Backbone(pretrained=pretrained)
        self.fc_mu = nn.Linear(self.encoder_backbone.model.last_linear.out_features, latent_dim)
        self.fc_var = nn.Linear(self.encoder_backbone.model.last_linear.out_features, latent_dim)

        '''
        Build Decoder
        '''
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
        
        self.decoder = nn.Sequential(*modules)

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
        
    def encode(self, x):
        x = self.encoder_backbone(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return [mu, logvar]
    
    def decode(self, z):
        '''
        Maps the given latent vector onto the image space
        '''
        z = self.decoder_input(z)
        z = self.decoder(z)
        z = self.final_layer(z)
        return z
    
    def reparameterize(self, mu, logvar):
        '''
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        '''
        std = logvar.mul(0.5).exp_()
        eps = torch.empty_like(std).normal_()
        return eps*std+mu
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            return mu
    
    def loss_function(self):
        