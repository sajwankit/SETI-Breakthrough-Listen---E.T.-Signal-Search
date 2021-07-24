import pandas as pd
import numpy as np
import torch
import warnings
import os
warnings.filterwarnings('ignore')
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)

import vae
import models
import config
import dataset
import seedandlog
import os
pre_classifier = models.get_model(pretrained=True, net_out_features=config.TARGET_SIZE)
class SaveFeatures():
    def __init__(self, m):
        """ Extract pretrained activations"""
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output):
        self.features = output.data.cpu().numpy()
    def remove(self):
        self.hook.remove()

class VAEPL(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=1024, classifier_features_dim=8192, input_height=256):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim+classifier_features_dim, 
            input_height=input_height, 
            first_conv=False, 
            maxpool1=False
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        
        # pretrained classifier for decoder input
        self.pre_classifier = models.get_model(pretrained=True, net_out_features=config.TARGET_SIZE)
        self.pre_classifier_features = SaveFeatures(self.pre_classifier._modules.get('prelu'))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
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

    def training_step(self, batch, batch_idx):
        x= batch['images']
        targets = batch['targets'] if batch['targets'] is not None else None

        xt = nn.functional.interpolate(x, size=(256, 256))
        xt = xt.repeat(1, 3, 1, 1)
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(xt)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        
        #pretrained classifier
        classifier_outputs = self.pre_classifier(x)
        
        z_ = torch.cat((z, classifier_outputs), dim=1)
        # decoded 
        x_hat = vae.decoder(z_)
        x_hat = nn.functional.interpolate(x_hat, size=(config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
        x_hat = nn.functional.instance_norm(x_hat)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(), 
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo